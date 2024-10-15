"""
Many imports
"""
from pylablib.devices import Thorlabs
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
import app2
import datetime
from statistics import mean, stdev
import math

import random
import matplotlib
from matplotlib import cm
import time

"""
Import files from local system
"""
dwf = ctypes.cdll.dwf

HERE = Path(__file__).parent
path.append(str(HERE / 'py/'))
path.append('py')
# import dwfconstants as constants
from WF_SDK import device, scope, wavegen, error   # import instruments

modulename = "WF_SDK"
if modulename not in sys.modules:
    print(f"You have not imported the {modulename} module")

class data:
    """ stores the sampling frequency and the buffer size for Digilent to use """
    sampling_frequency = 8.192e04
    buffer_size = 8192
    total_time = 0.1

class MotorApplication:
    """
    Handles motor control and spectrum observation.
    """
    def __enter__(self):
        """Allows for motor, digilent to be properly spun down and closed using __exit__ even if program is interrupted. """
        return self
    def __init__(self, currPeaked, serialNoYaw, serialNoPitch, yawBoundary, pitchBoundary, deviationVal, startYaw, startPitch, loc_log):
        """Initialize monitoring setup.

        Keyword arguments:
        serialNoYaw, serialNoPitch -- Serial numbers corresponding to each ThorLabs kdc101 controller.
        yawBoundary, pitchBoundary -- Hardcoded values beyond which the motor's limit switch will trigger.
        deviationVal -- The signal intensity percentage which PlotField uses as its field boundaries.
        startYaw, startPitch -- The initial homed yaw and pitch, reset when recenterHome is called. Note: not the motors' home values.
        loc_log -- Text file where timestamped motor positions are output after each move.
        """
        self.sample_size = 20
        self.binFactorYaw = 1
        self.binFactorPitch = 1
        # step_pitch = 4.88e-4
        self.step_pitch = 7.234e-6
        # step_yaw = 8.79e-4
        self.step_yaw = 13.023e-6


        self.deviationVal = deviationVal
        self.keepingCentered = False
        self.loc_log = loc_log
        self.window_revealed = False # GUI has not been set up yet, so don't display any movements

        """ Initialize motors and move them to manually input beam center."""
        self.deviceYaw = Thorlabs.KinesisMotor(serialNoYaw)
        self.devicePitch = Thorlabs.KinesisMotor(serialNoPitch)
        self.yawBoundary = yawBoundary
        self.pitchBoundary = pitchBoundary
        time.sleep(0.5)
        # print(self.deviceYaw.get_device_info())
        self.yaw_locs = []
        self.pitch_locs = []
        if not currPeaked:
            self.recenterHome(startYaw, startPitch)
        else:
            self.recenterHome(self.deviceYaw.get_position(), self.devicePitch.get_position())
        print(f'pos: {self.deviceYaw.get_position()}, {self.devicePitch.get_position()}')

        """ Run digilent """
        self.device_data = device.open()
        print("digilent device opened")
        scope.open(self.device_data, sampling_frequency=8.192e04, buffer_size=8192, amplitude_range=50)
        scope.trigger(self.device_data, enable=False, source=scope.trigger_source.analog, channel=1, level=0)
        print("scope set up")

        """Get a spectrum, and send it to GUI to display"""
        self.nyq_freq = data.sampling_frequency/2 # Nyquist frequency determines bounds of FFT
        buffer = scope.record(self.device_data, channel=1)
        print(f"recorded, len {len(buffer)}")
        self.total_samples = len(buffer)
        self.freq_range = np.linspace(0, self.nyq_freq, int(self.total_samples/2)+1)
        self.target_freq = 9e03
        self.target_index = self.target_freq*len(self.freq_range)/self.nyq_freq
        self.start_index = int(self.target_index*0.9)
        print(f'start index: {self.start_index}')
        self.end_index = int(self.target_index*1.1)
        print(f'end index: {self.end_index}')
        # self.spectrum = (np.abs(scipy.fft.rfft(buffer))*2/self.total_samples)
        self.freq_range = self.freq_range[self.start_index:self.end_index]
        # self.spectrum = self.spectrum[self.start_index:self.end_index]
        self.currVal, _ = self.checkSpectrum()

    def calibrateSampleBin(self):
        # self.calibrateSampleSize()
        # self.adjustBeams()
        # self.recenterHome(self.deviceYaw.get_position(), self.devicePitch.get_position())
        self.calibrateBinFactor()
        self.adjustBeams()
        
    def calibrateSampleSize(self):
        data_arr = []
        for i in range(1000):
            print(i)
            val = np.max(self.getSpectrum()[self.start_index:self.end_index])
            data_arr.append(val)
        print(f'average: {mean(data_arr)}, stdev: {stdev(data_arr)}')
        moe = mean(data_arr)/20
        zscore = 1.96
        n = (zscore*stdev(data_arr)/moe)**2
        self.sample_size = int(math.ceil(n))
        print(f'n is {n}')
    
    def calibrateBinFactor(self):
        self.binFactorYaw = 1
        self.binFactorPitch = 1
        hasMovedEnough = False
        self.currVal = self.checkVal()
        goingForwardYaw = 1
        print(f'going forward yaw: {goingForwardYaw}')
        while not hasMovedEnough:
            self.safeMove(self.deviceYaw, goingForwardYaw*self.binFactorYaw)
            newVal = self.checkCalibrationVal()
            self.binFactorYaw *= 2
            percent_diff = abs(newVal/self.currVal - 1)
            print(f'percent diff: {percent_diff}')
            hasMovedEnough = abs(newVal/self.currVal - 1) > .05
        self.binFactorYaw /= 2
        print(f'bin factor yaw: {self.binFactorYaw}')
        self.safeMove(self.deviceYaw, goingForwardYaw*(-self.binFactorYaw + 1))
        hasMovedEnough = False
        goingForwardPitch = 1
        print(f'going forward pitch: {goingForwardPitch}')
        while not hasMovedEnough:
            self.safeMove(self.devicePitch, goingForwardPitch*self.binFactorPitch)
            newVal = self.checkCalibrationVal()
            self.binFactorPitch *= 2
            percent_diff = abs(newVal/self.currVal - 1)
            print(f'percent diff: {percent_diff}')
            hasMovedEnough = abs(newVal/self.currVal - 1) > .05
        self.binFactorPitch /= 2
        print(f'bin factor pitch: {self.binFactorPitch}')
        self.safeMove(self.devicePitch, goingForwardPitch*(-self.binFactorPitch + 1))


    def windowRevealed(self, window):
        """ Called from GUI when it is ready to display. No updates will be made until the window is revealed. """
        self.main_window = window
        self.window_revealed = True

    def recenterHome(self, newYaw, newPitch):
        """ Recenter home and return home. """
        
        self.deviceYaw.setup_limit_switch(sw_kind='ignore') # Not using limit switches for testing, as they quietly stop the motor when the limit switch is reached, not sure how to tell when that happens.
        self.devicePitch.setup_limit_switch(sw_kind='ignore')
        # self.deviceYaw.setup_limit_switch(sw_kind='stop_imm', sw_position_cw = self.start_loc_yaw+yawBoundary, sw_position_ccw=self.start_loc_yaw-yawBoundary)
        # self.devicePitch.setup_limit_switch(sw_kind='stop_imm', sw_position_cw = self.start_loc_pitch+pitchBoundary, sw_position_ccw=self.start_loc_pitch-pitchBoundary)
        self.start_loc_yaw = newYaw
        self.start_loc_pitch = newPitch
        self.home()

    def getSpectrum(self):
        """ Get a single spectrum measurement from the digilent, then plot it if the GUI is revealed. """
        buffer = scope.record(self.device_data, channel=1)
        self.total_samples = len(buffer)
        spectrum = np.abs(scipy.fft.rfft(buffer))*2/self.total_samples
        self.export_spectrum = spectrum[self.start_index:self.end_index]
        # if self.window_revealed:
        #     self.main_window.update_plot1(spectrum[self.start_index:self.end_index])
        return spectrum

    def checkSpectrum(self):
        """ Run getSpectrum 10 times, average them and denoise, then extract the peak and return """
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
        graph_spectrum = spectrum[self.start_index:self.end_index]
        return np.max(denoised_spectrum[self.start_index:self.end_index]), graph_spectrum


    def checkVal(self):
        """ Just get the amplitude of the 9kHz modulation peak from checkSpectrum. """
        currValReturn, _ = self.checkSpectrum()
        self.globalVal = currValReturn
        return currValReturn
    
    def checkCalibrationVal(self):
        vals = []
        for _ in range(10):
            vals.append(self.checkVal())
        return mean(vals)

    def adjustBeams(self):
        """ Call adjustBeam on each of yaw and pitch separately (peak them up). """
        yawVal = self.adjustBeam(self.deviceYaw, self.binFactorYaw)
        pitchVal = self.adjustBeam(self.devicePitch, self.binFactorPitch)
        return yawVal, pitchVal

    def adjustBeam(self, device, startingBinFactor):
        """ Peak up the signal on the selected motor's axis. Algorithm described in Notion."""
        # self.currVal = self.checkVal()
        # newVal = self.currVal
        # goingForward = random.randint(0, 1)*2-1
        # print(f"flipped a {goingForward}")
        binFactor = startingBinFactor
        goingForward = self.findBetterDirection(device, startingBinFactor)
        binFactor *= 2
        if goingForward != 0:
            newVal = self.checkVal()
            while newVal >= self.currVal:
                self.currVal = newVal
                self.safeMove(device, goingForward*binFactor)
                binFactor *= 2
                newVal = self.checkVal()
            goingForward *= -1
            nextReverse = 1
            while binFactor > startingBinFactor:
                binFactor /= 2
                self.safeMove(device, goingForward*binFactor)
                newVal = self.checkVal()
                goingForward = nextReverse * goingForward
                if newVal > self.currVal:
                    self.currVal = newVal
                    nextReverse = -1
                else:
                    nextReverse = 1
            self.safeMove(device, binFactor*goingForward)
            newVal = self.checkVal()
            if newVal < self.currVal:
                goingForward *= nextReverse
                self.safeMove(device, binFactor*goingForward)

        newVal = self.checkVal()
        self.remember_location()
        return newVal
    
    def findBetterDirection(self, device, binFactor):
        self.currVal = self.checkVal()
        self.safeMove(device, binFactor)
        posVal = self.checkVal()
        self.safeMove(device, -2*binFactor)
        negVal = self.checkVal()
        self.safeMove(device, binFactor)
        if posVal < self.currVal and negVal < self.currVal:
            goingForward = 0
        elif posVal > negVal:
            goingForward = 1
        else:
            goingForward = -1
        self.safeMove(device, binFactor*goingForward)
        self.currVal = self.checkVal()
        return goingForward

    def checkSteps(self):
        """ Mostly unused function, checks how long it takes for each motor to go different distances."""
        totalIterations = 200

        watchYawStart = time.time()
        for i in range(totalIterations):
            self.safeMove(self.deviceYaw, 1)
        watchYawTotal = time.time() - watchYawStart
        print(f"Avg Yaw Time: {watchYawTotal/totalIterations}")

        watchPitchStart = time.time()
        for i in range(totalIterations):
            self.safeMove(self.devicePitch, 1)
        watchPitchTotal = time.time() - watchPitchStart
        print(f"Avg Pitch Time: {watchPitchTotal/totalIterations}")

        watchYawTenStart = time.time()
        for i in range(totalIterations):
            self.safeMove(self.deviceYaw, 10)
        watchYawTenTotal = time.time() - watchYawTenStart
        print(f"Avg 10x Yaw Time: {watchYawTenTotal/totalIterations}")

    def findBoundaries(self, device, binFactor):
        """ Before plotting the field, find the boundaries at which the signal drops below a specified value in both yaw and pitch.
        Go to extrema in all 4 directions, and return the number of steps of binFactor length required to get to those extrema.
        """
        self.currVal = self.checkVal()
        newVal = self.currVal
        offset_max = 0
        offset_min = 0
        starting_loc = device.get_position()
        print(starting_loc, self.start_loc_yaw, self.start_loc_pitch)
        while newVal >= self.currVal*self.deviationVal:
            self.safeMove(device, binFactor)
            print(device.get_position())
            time.sleep(0.5)
            newVal = self.checkVal()
            print(f'currval: {self.currVal}, newval: {newVal}')
            offset_max += 1
        print(f"started at {starting_loc}, now at {device.get_position()}")
        self.home()
        print(f"moved back to {device.get_position()}")
        newVal = self.checkVal()
        print(f"new val: {newVal}")
        # if 0.8 > newVal/self.currVal or newVal/self.currVal > 1.2:
        #     print("exception, your honor")
        #     raise Exception(f"motor hasn't returned sufficiently to origin going up, amplitude is now {newVal/self.currVal} of previous")
        self.currVal = newVal
        while newVal >= self.currVal*self.deviationVal:
            self.safeMove(device, -binFactor)
            print(device.get_position())
            time.sleep(0.5)
            newVal = self.checkVal()
            print(f'currval: {self.currVal}, newval: {newVal}')
            offset_min += 1
        self.home()
        newVal = self.checkVal()
        # if 0.8 > newVal/self.currVal or newVal/self.currVal > 1.2:
        #     print("exception, your honor")
        #     raise Exception(f"motor hasn't returned sufficiently to origin going down, amplitude is now {newVal/self.currVal} of previous")
        print(offset_min*binFactor, offset_max*binFactor)
        return offset_min*binFactor, offset_max*binFactor

    def getOrientationArray(self, minYaw, maxYaw, minPitch, maxPitch, binFactor):
        """Given yaw and pitch extrema from findBoundaries, step left to right, then top to bottom, graphin amplitude at each point."""
        self.safeMove(self.deviceYaw, maxYaw)
        self.safeMove(self.devicePitch, maxPitch)
        yawDiff = int((minYaw + maxYaw)/binFactor)
        pitchDiff = int((minPitch + maxPitch)/binFactor)
        amplitude_2d = np.ones((pitchDiff, yawDiff))
        print(maxYaw, maxPitch, minYaw, minPitch)
        for i in range (pitchDiff):
            self.safeMove(self.devicePitch, -binFactor)
            for j in range(yawDiff):
                self.safeMove(self.deviceYaw, -binFactor)
                amplitude_2d[i, j] = self.checkVal()
                print(i, j)
            self.safeMove(self.deviceYaw, yawDiff*binFactor)
        return amplitude_2d

    def safeMove(self, device, amount, velocity=79475.19421577454):
        """ A move function that also logs locations and displays in the GUI, then waits for the move to complete. """
        device.move_by(amount)
        device.wait_move()
        yaw_loc = self.deviceYaw.get_position()
        pitch_loc = self.devicePitch.get_position()
        self.yaw_locs.append(yaw_loc*self.step_yaw)
        self.pitch_locs.append(pitch_loc*self.step_pitch)
        # if self.window_revealed:
        #     self.main_window.update_plot2(self.yaw_locs, self.pitch_locs)
        # ct = datetime.datetime.now()
        # self.loc_log.write(f"{ct}, {yaw_loc*self.step_yaw}, {pitch_loc*self.step_pitch}, {self.globalVal},\n")

    def safeMoveTo(self, device, move_loc, velocity=79475.19421577454):
        curr_loc = device.get_position()
        self.safeMove(device, move_loc - curr_loc, velocity=velocity)

        
    def remember_location(self):
        yaw_loc = self.deviceYaw.get_position()
        pitch_loc = self.devicePitch.get_position()
        self.checkVal()
        ct = datetime.datetime.now()
        self.loc_log.write(f"{ct}, {yaw_loc*self.step_yaw}, {pitch_loc*self.step_pitch}, {self.globalVal},\n")

    def home(self, velocity=79475.19421577454):
        """Go to some predefined home."""
        print(self.start_loc_yaw)
        print(self.start_loc_pitch)
        self.deviceYaw.setup_velocity(max_velocity=velocity)
        self.devicePitch.setup_velocity(max_velocity=velocity)
        self.deviceYaw.move_to(self.start_loc_yaw)
        print("moooving")
        self.deviceYaw.wait_move()
        self.devicePitch.move_to(self.start_loc_pitch)
        self.devicePitch.wait_move()
        print(self.deviceYaw.get_position())
        print(self.devicePitch.get_position())
        print("home sweet home")
        time.sleep(2)
        yaw_loc = self.deviceYaw.get_position()
        pitch_loc = self.devicePitch.get_position()
        self.yaw_locs.append(yaw_loc*self.step_yaw)
        self.pitch_locs.append(pitch_loc*self.step_pitch)
        self.deviceYaw.setup_velocity(max_velocity=79475.19421577454)
        self.devicePitch.setup_velocity(max_velocity=79475.19421577454)
        # if self.window_revealed:
        #     self.main_window.update_plot2(self.yaw_locs, self.pitch_locs)

    def keepCentered(self):
        """Currently, run adjustBeams if the signal drops below a specific value, but could constantly adjust the beam."""
        self.keepingCentered = True
        while self.keepingCentered:
            newVal = self.checkVal()
            print(newVal, self.deviceYaw.get_position(), self.devicePitch.get_position())
            # if  newVal < self.currVal*self.deviationVal:
            self.currVal = self.adjustBeams()

    def plotField(self, binFactor):
        """ Peak up signal, then plot the field around the peak using findBoundaries and getOrientationArray """
        print(self.deviceYaw.get_scale_units())
        print(self.deviceYaw.get_scale())
        print(self.devicePitch.get_scale_units())
        print(self.devicePitch.get_scale())
        self.adjustBeams()
        newHomeYaw = self.deviceYaw.get_position()
        newHomePitch = self.devicePitch.get_position()
        print(f"yaw: {newHomeYaw}, pitch: {newHomePitch}")
        self.recenterHome(newHomeYaw, newHomePitch)
        minYaw, maxYaw = self.findBoundaries(self.deviceYaw, binFactor)
        print("yaw boundaries found")
        minPitch, maxPitch = self.findBoundaries(self.devicePitch, binFactor)
        print("pitch boundaries found")
        print(minYaw, maxYaw, minPitch, maxPitch)
        amplitude_2d = self.getOrientationArray(minYaw, maxYaw, minPitch, maxPitch, binFactor)
        plt.imshow(amplitude_2d, cmap=cm.coolwarm, interpolation='nearest', extent=[-minYaw*step_yaw, maxYaw*step_yaw, -minPitch*step_pitch, maxPitch*step_pitch])
        plt.colorbar()
        plt.xlabel("yaw extent (degrees)")
        plt.ylabel("pitch extent (degrees)")

        plt.savefig("spec_graph.png")
    
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Companion to __enter__. Safely shuts down each device whenever program is stopped/MotorApplication is closed."""
        self.deviceYaw.setup_limit_switch(sw_kind='ignore')
        self.devicePitch.setup_limit_switch(sw_kind='ignore')
        self.deviceYaw.stop()
        self.devicePitch.stop()
        self.deviceYaw.close()
        self.devicePitch.close()
        scope.close(self.device_data)
        device.close(self.device_data)
        motor_locs = zip(self.yaw_locs, self.pitch_locs)
        plt.ioff()
        with open('out.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(motor_locs)
        

# if __name__ == '__main__':
#     with open("loc_coords.txt", "ab") as loc_log:
#         # serialNoYaw = "27006315"
#         # serialNoPitch = "27006283"
#         serialNoYaw = "27250209"
#         serialNoPitch = "27250140"
#         yawBoundary = 2000
#         pitchBoundary = 1000
#         start_yaw = 218523
#         start_pitch = 241626
#         deviationVal = 0.5
#         binFactor = 50

#         with MotorApplication(serialNoYaw=serialNoYaw, serialNoPitch=serialNoPitch, yawBoundary=yawBoundary, pitchBoundary=pitchBoundary, deviationVal=deviationVal, startYaw=start_yaw, startPitch=start_pitch, loc_log=loc_log) as m:
#             print("about to plot field")
#             m.plotField(binFactor)