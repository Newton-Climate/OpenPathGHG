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

dwf = ctypes.cdll.dwf

# # load the dynamic library, get constants path (the path is OS specific)
# if platform.startswith("win"):
#     # on Windows
#     dwf = ctypes.cdll.dwf
#     constants_path = "C:" + sep + "Program Files (x86)" + sep + "Digilent" + sep + "WaveFormsSDK" + sep + "samples" + sep + "py"
# elif platform.startswith("darwin"):
#     # on macOS
#     lib_path = sep + "Library" + sep + "Frameworks" + sep + "dwf.framework" + sep + "dwf"
#     dwf = ctypes.cdll.LoadLibrary(lib_path)
#     constants_path = sep + "Applications" + sep + "WaveForms.app" + sep + "Contents" + sep + "Resources" + sep + "SDK" + sep + "samples" + sep + "py"
# else:
#     # on Linux
#     dwf = ctypes.cdll.LoadLibrary("libdwf.so")
#     constants_path = sep + "usr" + sep + "share" + sep + "digilent" + sep + "waveforms" + sep + "samples" + sep + "py"
 
# # import constants
# path.append(constants_path)
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
    def __init__(self, serialNoYaw, serialNoTilt, deviationVal):
        self.device_data = device.open()
        print("device opened")
        scope.open(self.device_data, sampling_frequency=8.192e04, buffer_size=8192, amplitude_range=50)
                # set up triggering on scope channel 1
        scope.trigger(self.device_data, enable=False, source=scope.trigger_source.analog, channel=1, level=0)

        print("scope set up")

        self.deviceYaw = Thorlabs.KinesisMotor(serialNoYaw)
        self.devicePitch = Thorlabs.KinesisMotor(serialNoTilt)
        time.sleep(0.5)
        print(self.deviceYaw.get_device_info())
        self.start_loc_yaw = self.deviceYaw.get_position()
        self.start_loc_pitch = self.devicePitch.get_position()

        # self.deviceYaw.home(sync=False, force=True, timeout=5000000)
        # self.devicePitch.home(sync=False, force=True, timeout=5000000)
        # print(self.deviceYaw.is_homing())
        # print(self.deviceYaw.is_homed())
        # self.deviceYaw.wait_for_home()
        # self.devicePitch.wait_for_home()
        # print("done homing")

        plt.ion()
        print("ion started")
        fig, (ax1, ax2) = plt.subplots(1, 2)
        buffer = scope.record(self.device_data, channel=1)
        print(f"recorded, len {len(buffer)}")
        self.total_samples = len(buffer)
        nyq_freq = data.sampling_frequency/2
        self.freq_range = np.linspace(0, nyq_freq, int(self.total_samples/2)+1)
        self.spectrum = np.abs(scipy.fft.rfft(buffer))*2/self.total_samples
        self.spec_graph = ax1.plot(self.freq_range[50:], self.spectrum[50:])[0]
        self.yaw_locs = [0]
        self.pitch_locs = [0]
        print("graph 1 done")
        self.loc_graph = ax2.plot(self.yaw_locs, self.pitch_locs)[0]
        plt.show()
        plt.pause(0.01)
        print("about to check spectrum")
        self.currVal, self.freq_range, self.spectrum = self.checkSpectrum()
        self.spec_graph.set_data(self.freq_range[50:], self.spectrum[50:])
        
        plt.pause(0.01)

        self.currPositionYaw = self.deviceYaw.get_position()
        self.currPositionPitch = self.devicePitch.get_position()
        self.deviationVal = deviationVal
        self.keepingCentered = False



    def checkSpectrum(self):
        nyq_freq = data.sampling_frequency/2
        freq_range = np.linspace(0, nyq_freq, int(self.total_samples/2)+1)
        
        target_freq = 9e03
        target_loc = target_freq*len(freq_range)/nyq_freq
        start_loc = int(target_loc*0.95)
        end_loc = int(target_loc*1.05)
        spectrums = []
        for _ in range(10):
            buffer = scope.record(self.device_data, channel=1)
            self.total_samples = len(buffer)
            spectrum = np.abs(scipy.fft.rfft(buffer))*2/self.total_samples
            self.spec_graph.set_ydata(spectrum[50:])
            
            plt.pause(0.01)
            spectrums.append(spectrum)
        spectrums = np.array(spectrums)
        spectrum = np.mean(spectrums, axis=0)
        return np.max(spectrum[start_loc:end_loc]), freq_range, spectrum


    def checkVal(self):
        self.currVal, _, _ = self.checkSpectrum()
        return self.currVal

    def adjustBeams(self):
        yawVal = self.adjustBeam(self.deviceYaw)
        pitchVal = self.adjustBeam(self.devicePitch)
        return yawVal, pitchVal

    def adjustBeam(self, device):
        self.currVal = self.checkVal(self.spec_graph)
        newVal = self.currVal
        goingForward = 1
        binFactor = 1
        while newVal >= self.currVal:
            self.currVal = newVal
            device.move_by(goingForward*binFactor)
            device.wait_move()
            binFactor *= 2
            newVal = self.checkVal()
        if binFactor == 2:
            binFactor = 1
            goingForward = -1
            device.move_by(goingForward*binFactor)
            device.wait_move()
            newVal = self.checkVal()
            self.currVal = newVal
            while newVal >= self.currVal:
                self.currVal = newVal
                device.move_by(goingForward*binFactor)
                device.wait_move()
                binFactor *= 2
                newVal = self.checkVal()

        goingForward *= -1
        nextReverse = 1
        while binFactor > 1:
            binFactor /= 2
            device.moveBy(goingForward*binFactor)
            newVal = self.checkVal()
            goingForward = nextReverse * goingForward
            if newVal > self.currVal:
                self.currVal = newVal
                nextReverse = -1
            else:
                nextReverse = 1
        device.move_by(binFactor*goingForward)
        device.wait_move()
        newVal = self.checkVal()
        if newVal < self.currVal:
            goingForward *= nextReverse
            device.move_by(binFactor*goingForward)
            device.wait_move()

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
        while newVal >= self.currVal*self.deviationVal:
            device.move_by(binFactor)
            print(device.get_position())
            newVal = self.checkVal()
            offset_max += 1
        device.move_by(-binFactor*offset_max)
        newVal = self.checkVal()
        if 0.9 < newVal/self.currVal < 1.1:
            raise Exception(f"motor hasn't returned sufficiently to origin going up, amplitude is now {newVal/self.currVal} of previous")
        self.currVal = newVal
        while newVal >= self.currVal*self.deviationVal:
            device.move_by(-binFactor)
            offset_min += 1
        device.move_by(binFactor*offset_min)
        newVal = self.checkVal()
        if 0.9 < newVal/self.currVal < 1.1:
            raise Exception(f"motor hasn't returned sufficiently to origin going down, amplitude is now {newVal/self.currVal} of previous")
        print(offset_min*binFactor, offset_max*binFactor)
        return offset_min*binFactor, offset_max*binFactor

    def getOrientationArray(self, minYaw, maxYaw, minPitch, maxPitch, binFactor):
        self.safeMove(self.deviceYaw, maxYaw)
        self.safeMove(self.devicePitch, maxPitch)
        yawDiff = (minYaw + maxYaw)/binFactor
        pitchDiff = (minPitch + maxPitch)/binFactor
        amplitude_2d = np.ones((pitchDiff, yawDiff))
        for i in range (pitchDiff):
            self.safeMove(self.devicePitch, -binFactor)
            for j in range(yawDiff):
                self.safeMove(self.deviceYaw, -binFactor)
                amplitude_2d[i, j] = self.checkVal()
            self.safeMove(self.deviceYaw, yawDiff*binFactor)
        return amplitude_2d

    def safeMove(self, device, amount):
        device.move_by(amount)
        while(device._is_moving()):
            yaw_loc = self.deviceYaw.get_position()
            pitch_loc = self.devicePitch.get_position()
            self.yaw_locs.append(yaw_loc)
            self.pitch_locs.append(pitch_loc)
            print(yaw_loc, pitch_loc)
            self.loc_graph.set_data(self.yaw_locs, self.pitch_locs)
            
            plt.pause(0.01)
            if np.abs(self.start_loc_yaw - yaw_loc) > 1000 or np.abs(self.start_loc_pitch - pitch_loc) > 500:
                print("stopping")
                print(self.yaw_locs, self.pitch_locs)
                self.deviceYaw.stop()
                self.devicePitch.stop()
                raise Exception(f"motor has left present boundaries with yaw of {yaw_loc} and pitch of {pitch_loc}")

    def keepCentered(self):
        self.keepingCentered = True
        while self.keepingCentered:
            newVal = self.checkVal()
            if  newVal < self.currVal*self.deviationVal:
                self.currVal = self.adjustBeams(self.deviceYaw, self.devicePitch)

    def plotField(self, binFactor):

        minYaw, maxYaw = self.findBoundaries(self.deviceYaw, binFactor)
        print("yaw boundaries found")
        minPitch, maxPitch = self.findBoundaries(self.devicePitch, binFactor)
        print("pitch boundaries found")
        amplitude_2d = self.getOrientationArray(minYaw, maxYaw, minPitch, maxPitch, self.deviceYaw, self.devicePitch)
        plt.imshow(amplitude_2d, cmap='hsv', interpolation='nearest')
        plt.savefig("spec_graph.png")
    
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
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
    serialNoYaw = "27006315"
    serialNoTilt = "27006283"
    deviationVal = 0.1
    binFactor = 50

    with MotorApplication(serialNoYaw, serialNoTilt, deviationVal) as m:
        print("about to plot field")
        m.plotField(binFactor)

    # except Exception as e:
    #     print("ERROR! ALERT!")
    #     print(e)