from pylablib.devices import Thorlabs
import ft232
import time
import ctypes                     # import the C compatible data types
from sys import platform, path    # this is needed to check the OS type and get the PATH
import sys
from pathlib import Path
from os import sep                # OS specific file path separators
import numpy as np
import scipy
import matplotlib.pyplot as plt
import csv
import app
import datetime

import random
import matplotlib
import time
matplotlib.use('QtAgg')

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QWidget, QPushButton

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

dwf = ctypes.cdll.dwf

HERE = Path(__file__).parent
path.append(str(HERE / 'py/'))
path.append('py')
import dwfconstants as constants
from WF_SDK import device, scope, wavegen, error   # import instruments

modulename = "WF_SDK"
if modulename not in sys.modules:
    print(f"You have not imported the {modulename} module")

class data:
    """ stores the sampling frequency and the buffer size """
    sampling_frequency = 8.192e04
    buffer_size = 8192
    total_time = 0.1

class MotorApplication:
    def __enter__(self):
        return self
    def __init__(self, serialNoYaw, serialNoPitch, yawBoundary, pitchBoundary, deviationVal, startYaw, startPitch, loc_log):
        self.device_data = device.open()
        print("device opened")
        scope.open(self.device_data, sampling_frequency=8.192e04, buffer_size=8192, amplitude_range=50)
                # set up triggering on scope channel 1
        scope.trigger(self.device_data, enable=False, source=scope.trigger_source.analog, channel=1, level=0)

        print("scope set up")
        self.window_revealed = False
        self.deviceYaw = Thorlabs.KinesisMotor(serialNoYaw)
        self.devicePitch = Thorlabs.KinesisMotor(serialNoPitch)
        self.yawBoundary = yawBoundary
        self.pitchBoundary = pitchBoundary
        time.sleep(0.5)
        print(self.deviceYaw.get_device_info())
        
        self.recenterHome(startYaw, startPitch)
        print(f'pos: {self.deviceYaw.get_position()}')

        self.nyq_freq = data.sampling_frequency/2
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

        
        self.spectrum = (np.abs(scipy.fft.rfft(buffer))*2/self.total_samples)
        self.freq_range = self.freq_range[self.start_index:self.end_index]
        self.spectrum = self.spectrum[self.start_index:self.end_index]
        
        self.currPositionYaw = self.deviceYaw.get_position()
        self.currPositionPitch = self.devicePitch.get_position()
        self.yaw_locs = [self.currPositionYaw]
        self.pitch_locs = [self.currPositionPitch]

        self.currVal, self.spectrum = self.checkSpectrum()

        self.deviationVal = deviationVal
        self.keepingCentered = False
        print(self.deviceYaw.get_limit_switch_parameters())

        self.loc_log = loc_log
            

    def windowRevealed(self, window):
        self.main_window = window
        print(self.main_window)
        self.window_revealed = True

    def recenterHome(self, newYaw, newPitch):
        self.deviceYaw.setup_limit_switch(sw_kind='ignore')
        self.devicePitch.setup_limit_switch(sw_kind='ignore')
        # self.deviceYaw.setup_limit_switch(sw_kind='stop_imm', sw_position_cw = self.start_loc_yaw+yawBoundary, sw_position_ccw=self.start_loc_yaw-yawBoundary)
        # self.devicePitch.setup_limit_switch(sw_kind='stop_imm', sw_position_cw = self.start_loc_pitch+pitchBoundary, sw_position_ccw=self.start_loc_pitch-pitchBoundary)

        self.start_loc_yaw = newYaw
        self.start_loc_pitch = newPitch
        self.deviceYaw.setup_velocity(max_velocity=79475.19421577454)
        self.deviceYaw.move_to(self.start_loc_yaw)
        print("waiting")
        self.deviceYaw.wait_move()
        self.devicePitch.setup_velocity(max_velocity=79475.19421577454)
        self.devicePitch.move_to(self.start_loc_pitch)
        print("waiting")
        self.devicePitch.wait_move()

    def getSpectrum(self):
        buffer = scope.record(self.device_data, channel=1)
        self.total_samples = len(buffer)
        spectrum = np.abs(scipy.fft.rfft(buffer))*2/self.total_samples
        if self.window_revealed:
            self.main_window.update_plot1(spectrum[self.start_index:self.end_index])
        return spectrum

    def checkSpectrum(self):

        spectrums = []
        for _ in range(10):
            spectrum = self.getSpectrum()
            spectrums.append(spectrum)
            
        spectrums = np.array(spectrums)
        spectrum = np.mean(spectrums, axis=0)
        noisy_spectrum = [*spectrum[:self.start_index], *spectrum[self.end_index:]]
        noise_level = np.mean(noisy_spectrum)
        denoised_spectrum = (spectrum - noise_level)
        graph_spectrum = spectrum[self.start_index:self.end_index]
        return np.max(denoised_spectrum[self.start_index:self.end_index]), graph_spectrum


    def checkVal(self):
        currValReturn, _ = self.checkSpectrum()
        return currValReturn

    def adjustBeams(self, binFactor):
        yawVal = self.adjustBeam(self.deviceYaw, binFactor)
        pitchVal = self.adjustBeam(self.devicePitch, binFactor)
        return yawVal, pitchVal

    def adjustBeam(self, device, binFactor):
        self.currVal = self.checkVal()
        newVal = self.currVal
        goingForward = 1
        while newVal >= self.currVal:
            self.currVal = newVal
            self.safeMove(device, goingForward*binFactor)
            binFactor *= 2
            newVal = self.checkVal()
        if binFactor == 2:
            binFactor = 1
            goingForward = -1
            self.safeMove(device, goingForward*binFactor)
            newVal = self.checkVal()
            self.currVal = newVal
            while newVal >= self.currVal:
                self.currVal = newVal
                self.safeMove(device, goingForward*binFactor)
                binFactor *= 2
                newVal = self.checkVal()

        goingForward *= -1
        nextReverse = 1
        while binFactor > 1:
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

        return newVal

    def checkSteps(self):
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
        self.currVal = self.checkVal()
        newVal = self.currVal
        offset_max = 0
        offset_min = 0
        starting_loc = device.get_position()
        print(starting_loc, self.start_loc_yaw, self.start_loc_pitch)
        while newVal >= self.currVal*self.deviationVal:
            self.safeMove(device, binFactor)
            print(device.get_position())
            newVal = self.checkVal()
            print(f'currval: {self.currVal}, newval: {newVal}')
            offset_max += 1
        print(f"started at {starting_loc}, now at {device.get_position()}")
        self.home(1000)
        print(f"moved back to {device.get_position()}")
        newVal = self.checkVal()
        print(f"new val: {newVal}")
        if 0.8 > newVal/self.currVal or newVal/self.currVal > 1.2:
            print("exception, your honor")
            raise Exception(f"motor hasn't returned sufficiently to origin going up, amplitude is now {newVal/self.currVal} of previous")
        self.currVal = newVal
        while newVal >= self.currVal*self.deviationVal:
            self.safeMove(device, -binFactor)
            print(device.get_position())
            newVal = self.checkVal()
            print(f'currval: {self.currVal}, newval: {newVal}')
            offset_min += 1
        self.home(1000)
        newVal = self.checkVal()
        if 0.8 > newVal/self.currVal or newVal/self.currVal > 1.2:
            print("exception, your honor")
            raise Exception(f"motor hasn't returned sufficiently to origin going down, amplitude is now {newVal/self.currVal} of previous")
        print(offset_min*binFactor, offset_max*binFactor)
        return offset_min*binFactor, offset_max*binFactor

    def getOrientationArray(self, minYaw, maxYaw, minPitch, maxPitch, binFactor):
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
        device.move_by(amount)
        device.wait_move()
        yaw_loc = self.deviceYaw.get_position()
        pitch_loc = self.devicePitch.get_position()
        self.yaw_locs.append(yaw_loc)
        self.pitch_locs.append(pitch_loc)
        if self.window_revealed:
            self.main_window.update_plot2(self.yaw_locs, self.pitch_locs)
        ct = datetime.datetime.now()
        self.loc_log.write(f"{ct}, {yaw_loc}, {pitch_loc},\n")
        
    def home(self, velocity=79475.19421577454):
        print(self.start_loc_yaw)
        print(self.start_loc_pitch)
        self.deviceYaw.setup_velocity(max_velocity=velocity)
        self.devicePitch.setup_velocity(max_velocity=velocity)
        self.deviceYaw.move_to(self.start_loc_yaw)
        self.deviceYaw.wait_move()
        self.devicePitch.move_to(self.start_loc_pitch)
        self.devicePitch.wait_move()
        print(self.deviceYaw.get_position())
        print(self.devicePitch.get_position())
        print("home sweet home")
        time.sleep(2)
        yaw_loc = self.deviceYaw.get_position()
        pitch_loc = self.devicePitch.get_position()
        self.yaw_locs.append(yaw_loc)
        self.pitch_locs.append(pitch_loc)
        self.deviceYaw.setup_velocity(max_velocity=79475.19421577454)
        self.devicePitch.setup_velocity(max_velocity=79475.19421577454)
        if self.window_revealed:
            self.main_window.update_plot2(self.yaw_locs, self.pitch_locs)


    def keepCentered(self, binFactor):
        self.keepingCentered = True
        while self.keepingCentered:
            newVal = self.checkVal()
            if  newVal < self.currVal*self.deviationVal:
                self.currVal = self.adjustBeams(self.deviceYaw, self.devicePitch, binFactor)

    def plotField(self, binFactor):
        print(self.deviceYaw.get_scale_units())
        print(self.deviceYaw.get_scale())
        print(self.devicePitch.get_scale_units())
        print(self.devicePitch.get_scale())
        self.adjustBeams(1)
        newHomeYaw = self.deviceYaw.get_position()
        newHomePitch = self.devicePitch.get_position()
        print(f"yaw: {newHomeYaw}, pitch: {newHomePitch}")
        self.recenterHome(newHomeYaw, newHomePitch)
        minYaw, maxYaw = self.findBoundaries(self.deviceYaw, binFactor)
        print("yaw boundaries found")
        # self.main_window.quitApp()
        minPitch, maxPitch = self.findBoundaries(self.devicePitch, binFactor)
        print("pitch boundaries found")
        print(minYaw, maxYaw, minPitch, maxPitch)
        amplitude_2d = self.getOrientationArray(minYaw, maxYaw, minPitch, maxPitch, binFactor)
        step_pitch = 4.88e-4
        step_yaw = 8.79e-4
        plt.imshow(amplitude_2d, cmap='hsv', interpolation='nearest', extent=[-minYaw*step_yaw, maxYaw*step_yaw, -minPitch*step_pitch, maxPitch*step_pitch])
        plt.colorbar()

        plt.savefig("spec_graph.png")
    
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
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
        

if __name__ == '__main__':
    with open("loc_coords.txt", "ab") as loc_log:
        # serialNoYaw = "27006315"
        # serialNoPitch = "27006283"
        serialNoYaw = "27250209"
        serialNoPitch = "27250140"
        yawBoundary = 2000
        pitchBoundary = 1000
        start_yaw = 218523
        start_pitch = 241626
        deviationVal = 0.5
        binFactor = 50

        with MotorApplication(serialNoYaw=serialNoYaw, serialNoPitch=serialNoPitch, yawBoundary=yawBoundary, pitchBoundary=pitchBoundary, deviationVal=deviationVal, startYaw=start_yaw, startPitch=start_pitch, loc_log=loc_log) as m:
            print("about to plot field")
            m.plotField(binFactor)