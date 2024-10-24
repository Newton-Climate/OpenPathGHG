"""
Many imports
"""

import ft232
import time
import ctypes
from sys import platform, path
import sys
from pathlib import Path
from os import sep
import numpy as np
import scipy
import matplotlib.pyplot as plt
import csv
import app
import datetime
from statistics import mean, stdev
import math
import copy

import random
import matplotlib
from matplotlib import cm
import time

import digilent_scope, thorlabs_motor


class BeamAligner:
    """
    Handles motor control and spectrum observation.
    """

    def __enter__(self):
        """Allows for motor, digilent to be properly spun down and closed using __exit__ even if program is interrupted."""
        return self

    def __init__(
        self,
        currPeaked,
        serialNoYaw,
        serialNoPitch,
        yawBoundary,
        pitchBoundary,
        deviationVal,
        startYaw,
        startPitch,
        loc_log,
        fogThreshold,
    ):
        """Initialize a BeamAligner object, along with its motor and scope controller.

        Args:
            currPeaked (bool): If false, move to the startYaw and startPitch and use those as home. If true, use current location as home.
            serialNoYaw (int): Serial numbers corresponding to each ThorLabs kdc101 controller.
            serialNoPitch (int): Serial numbers corresponding to each ThorLabs kdc101 controller.
            yawBoundary (int): Hardcoded values beyond which the motor's limit switch will trigger.
            pitchBoundary (int): Hardcoded values beyond which the motor's limit switch will trigger.
            deviationVal (float): The signal intensity percentage which PlotField uses as its field boundaries.
            startYaw (int): The initial homed yaw and pitch, reset when recenterHome is called. Note: not the motors' home values.
            startPitch (int): The initial homed yaw and pitch, reset when recenterHome is called. Note: not the motors' home values.
            loc_log (.txt file): Text file where timestamped motor positions are output after each move. Opened in app.py.
        """
        self.fogThreshold = fogThreshold
        self.sample_size = 20
        self.binFactorYaw = 1
        self.binFactorPitch = 1
        """NOTE: step_yaw and step_pitch do not have to do with the binFactor, these are conversion factors from internal motor steps->degrees moved."""
        self.step_pitch = 7.234e-6
        self.step_yaw = 13.023e-6

        self.deviationVal = deviationVal
        self.keepingCentered = False
        self.loc_log = loc_log
        self.window_revealed = (
            False  # GUI has not been set up yet, so don't display any movements
        )

        """ Initialize motors and move them to manually input beam center."""
        self.motor_controller = thorlabs_motor.Motor(
            serialNoYaw=serialNoYaw,
            serialNoPitch=serialNoPitch,
            step_yaw=self.step_yaw,
            step_pitch=self.step_pitch,
        )

        if not currPeaked:
            self.recenterHome(startYaw, startPitch)
        else:
            self.recenterHome(
                self.motor_controller.getPosition(deviceName="yaw"),
                self.motor_controller.getPosition(deviceName="pitch"),
            )
        print(
            f"pos: {self.motor_controller.getPosition(deviceName="yaw")}, {self.motor_controller.getPosition(deviceName="pitch")}"
        )

        self.scope_controller = digilent_scope.Scope()

        """Get a spectrum, and send it to GUI to display"""
        self.nyq_freq = (
            data.sampling_frequency / 2
        )  # Nyquist frequency determines bounds of FFT
        buffer = self.scope_controller.recordData()
        print(f"recorded, len {len(buffer)}")
        self.total_samples = len(buffer)
        self.freq_range = np.linspace(0, self.nyq_freq, int(self.total_samples / 2) + 1)
        self.target_freq = 9e03
        self.target_index = self.target_freq * len(self.freq_range) / self.nyq_freq
        self.start_index = int(self.target_index * 0.9)
        print(f"start index: {self.start_index}")
        self.end_index = int(self.target_index * 1.1)
        print(f"end index: {self.end_index}")
        self.freq_range = self.freq_range[self.start_index : self.end_index]
        currVal, _ = self.checkSpectrum()
        self.initialPeakVal = currVal
        self.valBuffer = [(0, 0, 0)] * 50
        self.addToValBuffer(
            self.initialPeakVal,
            self.motor_controller.getPosition(deviceName="yaw"),
            self.motor_controller.getPosition(deviceName="pitch"),
        )
        print(self.valBuffer)

    def addToValBuffer(self, amplitude, loc_yaw, loc_pitch):
        """Save a historically measured value to a buffer of previous values, and delete the oldest one.

        Args:
            amplitude (float): The amplitude of signal measured.
            loc_yaw (int): Yaw location of this measurement.
            loc_pitch (int): Pitch location of this measurement.
        """
        self.valBuffer.pop(0)
        self.valBuffer.append((amplitude, loc_yaw, loc_pitch))

    def getMaxBuffer(self):
        """Return the amplitude, yaw, and pitch of the highest amplitude measurement in the buffer.

        Returns:
            tuple(int, int, int): Amplitude, yaw, and pitch.
        """
        return max(self.valBuffer, key=lambda valTuple: valTuple[0])

    def calibrateSampleBin(self):
        """Calibrate the sample size of each checkVal call, and then how big a step the motors should take in both the yaw and pitch directions. Peak up beam in between."""
        self.calibrateSampleSize()
        self.adjustBeams()
        self.recenterHome(
            self.motor_controller.getPosition(deviceName="yaw"),
            self.motor_controller.getPosition(deviceName="pitch"),
        )
        self.calibrateBinFactor()
        self.adjustBeams()

    def calibrateSampleSize(self):
        """Take 1000 data points, then assign the number of samples per measurement such that the value will be within moe margin of error, with uncertainty based on zscore.
        """        
        data_arr = []
        for i in range(1000):
            print(i)
            val = np.max(self.getSpectrum()[self.start_index : self.end_index])
            data_arr.append(val)
        print(f"average: {mean(data_arr)}, stdev: {stdev(data_arr)}")
        moe = mean(data_arr) / 20
        zscore = 1.96
        n = (zscore * stdev(data_arr) / moe) ** 2
        self.sample_size = int(math.ceil(n))
        print(f"n is {n}")

    def calibrateBinFactor(self):
        """Move in the yaw and pitch until the signal drops off sufficiently, doubling moves each time, then use that to assign the smallest motor step (binFactor) in yaw and pitch.
        """        
        self.binFactorYaw = 1
        self.binFactorPitch = 1
        hasMovedEnough = False
        currVal = self.checkVal()
        starting_loc_yaw = self.motor_controller.getPosition(deviceName="yaw")
        starting_loc_pitch = self.motor_controller.getPosition(deviceName="pitch")
        goingForwardYaw = 1
        print(f"going forward yaw: {goingForwardYaw}")
        while not hasMovedEnough:
            self.motor_controller.safeMove(
                deviceName="yaw", amount=goingForwardYaw * self.binFactorYaw
            )
            newVal = self.checkCalibrationVal()
            self.binFactorYaw *= 2
            percent_diff = abs(newVal / currVal - 1)
            print(f"percent diff: {percent_diff}")
            hasMovedEnough = abs(newVal / currVal - 1) > 0.05
        self.binFactorYaw /= 2
        print(f"bin factor yaw: {self.binFactorYaw}")
        self.safeMoveTo(deviceName="yaw", move_loc=starting_loc_yaw)
        hasMovedEnough = False
        goingForwardPitch = 1
        print(f"going forward pitch: {goingForwardPitch}")
        while not hasMovedEnough:
            self.motor_controller.safeMove(
                deviceName="pitch", amount=goingForwardPitch * self.binFactorPitch
            )
            newVal = self.checkCalibrationVal()
            self.binFactorPitch *= 2
            percent_diff = abs(newVal / currVal - 1)
            print(f"percent diff: {percent_diff}")
            hasMovedEnough = abs(newVal / currVal - 1) > 0.05
        self.binFactorPitch /= 2
        print(f"bin factor pitch: {self.binFactorPitch}")
        self.safeMoveTo(deviceName="pitch", move_loc=starting_loc_pitch)

    def windowRevealed(self, window):
        """Called from GUI when it is ready to display. No updates will be made until the window is revealed.

        Args:
            window (MainWindow): The MainWindow object that has been revealed by app.py.
        """
        self.main_window = window
        self.window_revealed = True

    def recenterHome(self, newYaw, newPitch):
        """Recenter home and return home.

        Args:
            newYaw (int): New home location yaw.
            newPitch (int): New home location pitch.
        """        

        self.motor_controller.setupLimitSwitch(
            deviceName="yaw", sw_kind="ignore"
        )  # Not using limit switches for testing, as they quietly stop the motor when the limit switch is reached, not sure how to tell when that happens.
        self.motor_controller.setupLimitSwitch(deviceName="pitch", sw_kind="ignore")
        # self.motor_controller.setupLimitSwitch(deviceName="yaw", sw_kind='stop_imm', sw_position_cw = self.start_loc_yaw+yawBoundary, sw_position_ccw=self.start_loc_yaw-yawBoundary)
        # self.motor_controller.setupLimitSwitch(deviceName="pitch", sw_position_cw = self.start_loc_pitch+pitchBoundary, sw_position_ccw=self.start_loc_pitch-pitchBoundary)
        self.start_loc_yaw = newYaw
        self.start_loc_pitch = newPitch
        self.home()

    def getSpectrum(self):
        """Get a single spectrum measurement from the digilent, then plot it if the GUI is revealed.

        Returns:
            list[float]: A spectrum of frequencies up th the Nyquist frequency
        """        
        buffer = self.scope_controller.recordData()
        self.total_samples = len(buffer)
        spectrum = np.abs(scipy.fft.rfft(buffer)) * 2 / self.total_samples
        self.export_spectrum = spectrum[self.start_index : self.end_index]
        return spectrum

    def checkSpectrum(self):
        """Run getSpectrum 10 times, average them and denoise (NOTE: NOT CURRENTLY DENOISING), then extract the peak and return.

        Returns:
            tuple(float, list[float]): The maximum spectrum amplitude near 9 kHz, and the remainder of the spectrum near 9 kHz.
        """        
        spectrums = []
        for _ in range(self.sample_size):
            spectrum = self.getSpectrum()
            spectrums.append(spectrum)

        spectrums = np.array(spectrums)
        spectrum = np.mean(spectrums, axis=0)
        # noisy_spectrum = [*spectrum[:self.start_index], *spectrum[self.end_index:]]
        # noise_level = np.mean(noisy_spectrum)
        # denoised_spectrum = (spectrum - noise_level) #Note: Could do more sophisticated denoising, right now just subtracting avg background
        denoised_spectrum = spectrum
        graph_spectrum = spectrum[self.start_index : self.end_index]
        return (
            np.max(denoised_spectrum[self.start_index : self.end_index]),
            graph_spectrum,
        )

    def checkVal(self):
        """Just get the amplitude of the 9kHz modulation peak from checkSpectrum.

        Returns:
            float: 9kHz modulation peak amplitude.
        """        
        currValReturn, _ = self.checkSpectrum()
        self.globalVal = currValReturn
        return currValReturn

    def checkCalibrationVal(self):
        """A version of checkVal which reads more signal peaks and averages them (slower, but more reliable value).

        Returns:
            float: Signal amplitude at 9kHz.
        """        
        vals = []
        for _ in range(10):
            vals.append(self.checkVal())
        return mean(vals)

    def waitForFog(self):
        """When fog is detected, this is called. Stop moving until the signal increases, then move to the highest amplitude location in the buffer.
        """        
        currVal = self.checkVal()
        while currVal < self.fogThreshold * self.initialPeakVal and self.keepingCentered:
            currVal = self.checkVal()
            time.sleep(20)
        self.safeMoveTo(deviceName="yaw", move_loc=self.getMaxBuffer()[1])
        self.safeMoveTo(deviceName="pitch", move_loc=self.getMaxBuffer()[2])

    def adjustBeams(self):
        """Call adjustBeam on each of yaw and pitch separately (peak them up). Then check whether there is a better place to move in the buffer, and move there if so.
        """        
        self.adjustBeam(deviceName="yaw", startingBinFactor=self.binFactorYaw)
        self.adjustBeam(deviceName="pitch", startingBinFactor=self.binFactorPitch)
        newVal = self.checkVal()
        newYaw = self.motor_controller.getPosition(deviceName="yaw")
        newPitch = self.motor_controller.getPosition(deviceName="pitch")
        self.addToValBuffer(newVal, newYaw, newPitch)
        maxVal, maxYaw, maxPitch = self.getMaxBuffer()
        self.safeMoveTo(deviceName="yaw", move_loc=maxYaw)
        self.safeMoveTo(deviceName="pitch", move_loc=maxPitch)
        if self.checkVal() < newVal:
            self.safeMoveTo(deviceName="yaw", move_loc=newYaw)
            self.safeMoveTo(deviceName="pitch", move_loc=newPitch)
        self.remember_location()

    def adjustBeam(self, deviceName, startingBinFactor):
        """Peak up the signal on the selected motor's axis. Algorithm described in Notion. Afterwards, move to the location with the highest amplitude signal seen during the algorithm's movement.

        Args:
            deviceName (str): "yaw" or "pitch", fed to the Motor.
            startingBinFactor (int): Starting copy of how many motor steps each step of the controller will take (minimum resolution)
        """        
        binFactor = copy.copy(startingBinFactor)
        goingForward = self.findBetterDirection(deviceName, startingBinFactor)
        max_loc = self.motor_controller.getPosition(deviceName=deviceName)
        currVal = self.checkVal()

        if goingForward != 0:
            self.motor_controller.safeMove(
                deviceName=deviceName, amount=goingForward * binFactor
            )
            newVal = self.checkVal()
            while newVal >= currVal:
                currVal = newVal
                max_loc = self.motor_controller.getPosition(deviceName=deviceName)
                self.motor_controller.safeMove(
                    deviceName=deviceName, amount=goingForward * binFactor
                )
                binFactor *= 2
                newVal = self.checkVal()
            goingForward *= -1
            nextReverse = 1
            binFactor /= 2
            while binFactor > startingBinFactor:
                binFactor /= 2
                self.motor_controller.safeMove(
                    deviceName=deviceName, amount=goingForward * binFactor
                )
                newVal = self.checkVal()
                goingForward = nextReverse * goingForward
                if newVal > currVal:
                    currVal = newVal
                    nextReverse = -1
                    max_loc = self.motor_controller.getPosition(deviceName=deviceName)
                else:
                    nextReverse = 1
            self.motor_controller.safeMove(
                deviceName=deviceName, amount=binFactor * goingForward
            )
            newVal = self.checkVal()
            if newVal < currVal:
                binFactor /= 2
                goingForward *= nextReverse
                self.motor_controller.safeMove(
                    deviceName=deviceName, amount=binFactor * goingForward
                )
            # Check if we should move back to greatest magnitude position
            newVal = self.checkVal()
            if newVal > currVal:
                max_loc = self.motor_controller.getPosition(deviceName=deviceName)
        self.safeMoveTo(deviceName=deviceName, move_loc=max_loc)

    def findBetterDirection(self, deviceName, binFactor):
        """When running adjust beam, instead of choosing a direction uniformly or randomly as previously to move first, instead move in both directions and choose the one with greater increase.

        Args:
            deviceName (str): "yaw" or "pitch", fed to the Motor.
            binFactor (int): How many motor steps each step of the controller will take (minimum resolution).

        Returns:
            int: Direction to go with higher amplitude signal.
        """        
        currVal = self.checkVal()
        self.motor_controller.safeMove(deviceName=deviceName, amount=binFactor)
        posVal = self.checkVal()
        self.motor_controller.safeMove(deviceName=deviceName, amount=-2 * binFactor)
        negVal = self.checkVal()
        self.motor_controller.safeMove(deviceName=deviceName, amount=binFactor)
        if posVal < currVal and negVal < currVal:
            goingForward = 0
        elif posVal > negVal:
            goingForward = 1
        else:
            goingForward = -1
        self.motor_controller.safeMove(
            deviceName=deviceName, amount=binFactor * goingForward
        )
        currVal = self.checkVal()
        return goingForward

    def checkSteps(self):
        """Mostly unused function, checks how long it takes for each motor to go different distances.
        """        
        totalIterations = 200

        watchYawStart = time.time()
        for i in range(totalIterations):
            self.motor_controller.safeMove(deviceName="yaw", amount=1)
        watchYawTotal = time.time() - watchYawStart
        print(f"Avg Yaw Time: {watchYawTotal/totalIterations}")

        watchPitchStart = time.time()
        for i in range(totalIterations):
            self.motor_controller.safeMove(deviceName="pitch", amount=1)
        watchPitchTotal = time.time() - watchPitchStart
        print(f"Avg Pitch Time: {watchPitchTotal/totalIterations}")

        watchYawTenStart = time.time()
        for i in range(totalIterations):
            self.motor_controller.safeMove(deviceName="yaw", amount=10)
        watchYawTenTotal = time.time() - watchYawTenStart
        print(f"Avg 10x Yaw Time: {watchYawTenTotal/totalIterations}")

    def findBoundaries(self, deviceName, binFactor):
        """Before plotting the field, find the boundaries at which the signal drops below a specified value in both yaw and pitch.
        Go to extrema in all 4 directions, and return the number of steps of binFactor length required to get to those extrema.

        Args:
            device (KinesisMotor): Either a yaw or pitch motor which will be adjusted.
            binFactor (int): How many motor steps each step of the controller will take (minimum resolution).

        Returns:
            int, int: the number of steps left and right to reach the boundaries of the field.
        """
        currVal = self.checkVal()
        newVal = currVal
        offset_max = 0
        offset_min = 0
        starting_loc = self.motor_controller.getPosition(deviceName=deviceName)
        print(starting_loc, self.start_loc_yaw, self.start_loc_pitch)
        while newVal >= currVal * self.deviationVal:
            self.motor_controller.safeMove(deviceName=deviceName, amount=binFactor)
            print(self.motor_controller.getPosition(deviceName=deviceName))
            time.sleep(0.5)
            newVal = self.checkVal()
            print(f"currval: {currVal}, newval: {newVal}")
            offset_max += 1
        print(
            f"started at {starting_loc}, now at {self.motor_controller.getPosition(deviceName=deviceName)}"
        )
        self.home()
        print(
            f"moved back to {self.motor_controller.getPosition(deviceName=deviceName)}"
        )
        newVal = self.checkVal()
        print(f"new val: {newVal}")
        currVal = newVal
        while newVal >= currVal * self.deviationVal:
            self.motor_controller.safeMove(deviceName=deviceName, amount=-binFactor)
            print(self.motor_controller.getPosition(deviceName=deviceName))
            time.sleep(0.5)
            newVal = self.checkVal()
            print(f"currval: {currVal}, newval: {newVal}")
            offset_min += 1
        self.home()
        newVal = self.checkVal()
        print(offset_min * binFactor, offset_max * binFactor)
        return offset_min * binFactor, offset_max * binFactor

    def getOrientationArray(self, minYaw, maxYaw, minPitch, maxPitch, binFactor):
        """Given yaw and pitch extrema from findBoundaries, step left to right, then top to bottom, graphing amplitude at each point.

        Args:
            minYaw (int): Minimum yaw boundary.
            maxYaw (int): Maximum yaw boundary.
            minPitch (int): Minimum pitch boundary.
            maxPitch (int): Maximum pitch boundary.
            binFactor (int): How many motor steps each step of the controller will take (minimum resolution).

        Returns:
            2d list[int]: Signal amplitudes at each coordinate of the mapped grid.
        """        
        self.motor_controller.safeMove(deviceName="yaw", amount=maxYaw)
        self.motor_controller.safeMove(deviceName="pitch", amount=maxPitch)
        yawDiff = int((minYaw + maxYaw) / binFactor)
        pitchDiff = int((minPitch + maxPitch) / binFactor)
        amplitude_2d = np.ones((pitchDiff, yawDiff))
        print(maxYaw, maxPitch, minYaw, minPitch)
        for i in range(pitchDiff):
            self.motor_controller.safeMove(deviceName="pitch", amount=-binFactor)
            for j in range(yawDiff):
                self.motor_controller.safeMove(deviceName="yaw", amount=-binFactor)
                amplitude_2d[i, j] = self.checkVal()
                print(i, j)
            self.motor_controller.safeMove(deviceName="yaw", amount=yawDiff * binFactor)
        return amplitude_2d

    def safeMoveTo(self, deviceName, move_loc, velocity=79475.19421577454):
        """Helper function that calls safeMove to a specific location instead of a relative value.

        Args:
            deviceName (str): "yaw" or "pitch", fed to the Motor.
            move_loc (int): Location to move the deviceName to.
            velocity (float, optional): How fast the motor should move. If the motor starts moving very slowly, check that it isn't stuck at a low value. Defaults to 79475.19421577454.
        """        
        curr_loc = self.motor_controller.getPosition(deviceName=deviceName)
        self.motor_controller.safeMove(
            deviceName=deviceName, amount=move_loc - curr_loc, velocity=velocity
        )

    def remember_location(self):
        """Save the current location and signal amplitude to loc_coords.txt in the data folder.
        """        
        yaw_loc = self.motor_controller.getPosition(deviceName="yaw")
        pitch_loc = self.motor_controller.getPosition(deviceName="pitch")
        self.checkVal()
        ct = datetime.datetime.now()
        self.loc_log.write(
            f"{ct}, {yaw_loc*self.step_yaw}, {pitch_loc*self.step_pitch}, {self.globalVal},\n"
        )

    def home(self, velocity=79475.19421577454):
        """Go to some predefined home.

        Args:
            velocity (float, optional): How fast the motor should move. If the motor starts moving very slowly, check that it isn't stuck at a low value. Defaults to 79475.19421577454.
        """        
        print(self.start_loc_yaw)
        print(self.start_loc_pitch)
        self.motor_controller.setupVelocity(deviceName="yaw", max_velocity=velocity)
        self.motor_controller.setupVelocity(deviceName="pitch", max_velocity=velocity)
        self.safeMoveTo(deviceName="yaw", move_loc=self.start_loc_yaw)
        self.safeMoveTo(deviceName="pitch", move_loc=self.start_loc_pitch)
        loc_yaw = self.motor_controller.getPosition(deviceName="yaw")
        loc_pitch = self.motor_controller.getPosition(deviceName="pitch")
        print(loc_yaw, loc_yaw * self.step_yaw)
        print(loc_pitch, loc_pitch * self.step_pitch)
        print("home sweet home")
        time.sleep(2)
        yaw_loc = self.motor_controller.getPosition(deviceName="yaw")
        pitch_loc = self.motor_controller.getPosition(deviceName="pitch")
        self.yaw_locs.append(yaw_loc * self.step_yaw)
        self.pitch_locs.append(pitch_loc * self.step_pitch)
        self.motor_controller.setupVelocity(
            deviceName="yaw", max_velocity=79475.19421577454
        )
        self.motor_controller.setupVelocity(
            deviceName="pitch", max_velocity=79475.19421577454
        )

    def keepCentered(self):
        """Continually run adjustBeams until fog occludes the beam too much.
        """        
        self.keepingCentered = True
        while self.keepingCentered:
            newVal = self.checkVal()
            if newVal < self.fogThreshold * self.initialPeakVal:
                self.waitForFog()
            print(
                newVal,
                self.motor_controller.getPosition(deviceName="yaw") * self.step_yaw,
                self.motor_controller.getPosition(deviceName="pitch") * self.step_pitch,
            )
            self.adjustBeams()

    def plotField(self, binFactor):
        """Peak up signal, then plot the field around the peak using findBoundaries and getOrientationArray

        Args:
            binFactor (int): How many motor steps each step of the controller will take (minimum resolution).
        """        
        print(self.motor_controller.getScaleUnits(deviceName="yaw"))
        print(self.motor_controller.getScale(deviceName="yaw"))
        print(self.motor_controller.getScaleUnits(deviceName="pitch"))
        print(self.motor_controller.getScale(deviceName="pitch"))
        self.adjustBeams()
        newHomeYaw = self.motor_controller.getPosition(deviceName="yaw")
        newHomePitch = self.motor_controller.getPosition(deviceName="pitch")
        print(f"yaw: {newHomeYaw}, pitch: {newHomePitch}")
        self.recenterHome(newHomeYaw, newHomePitch)
        minYaw, maxYaw = self.findBoundaries(deviceName="yaw", binFactor=binFactor)
        print("yaw boundaries found")
        minPitch, maxPitch = self.findBoundaries(
            deviceName="pitch", binFactor=binFactor
        )
        print("pitch boundaries found")
        print(minYaw, maxYaw, minPitch, maxPitch)
        amplitude_2d = self.getOrientationArray(
            minYaw, maxYaw, minPitch, maxPitch, binFactor
        )
        plt.imshow(
            amplitude_2d,
            cmap=cm.coolwarm,
            interpolation="nearest",
            extent=[
                -minYaw * self.step_yaw,
                maxYaw * self.step_yaw,
                -minPitch * self.step_pitch,
                maxPitch * self.step_pitch,
            ],
        )
        plt.colorbar()
        plt.xlabel("yaw extent (degrees)")
        plt.ylabel("pitch extent (degrees)")

        plt.savefig("spec_graph.png")

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Companion to __enter__. Safely shuts down each device whenever program is stopped/BeamAligner is closed."""
        self.motor_controller.shutDown()
        self.scope_controller.shutDown()
