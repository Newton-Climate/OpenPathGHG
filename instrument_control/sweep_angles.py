from pylablib.devices import Thorlabs
import ft232
import time
import ctypes                     # import the C compatible data types
from sys import platform, path    # this is needed to check the OS type and get the PATH
from os import sep                # OS specific file path separators
import numpy as np
import scipy
import matplotlib as plt

# load the dynamic library, get constants path (the path is OS specific)
if platform.startswith("win"):
    # on Windows
    dwf = ctypes.cdll.dwf
    constants_path = "C:" + sep + "Program Files (x86)" + sep + "Digilent" + sep + "WaveFormsSDK" + sep + "samples" + sep + "py"
elif platform.startswith("darwin"):
    # on macOS
    lib_path = sep + "Library" + sep + "Frameworks" + sep + "dwf.framework" + sep + "dwf"
    dwf = ctypes.cdll.LoadLibrary(lib_path)
    constants_path = sep + "Applications" + sep + "WaveForms.app" + sep + "Contents" + sep + "Resources" + sep + "SDK" + sep + "samples" + sep + "py"
else:
    # on Linux
    dwf = ctypes.cdll.LoadLibrary("libdwf.so")
    constants_path = sep + "usr" + sep + "share" + sep + "digilent" + sep + "waveforms" + sep + "samples" + sep + "py"
 
# import constants
path.append(constants_path)
import dwfconstants as constants
from WF_SDK import device, scope, wavegen, error   # import instruments

class data:
    """ stores the sampling frequency and the buffer size """
    sampling_frequency = 8.192e04
    buffer_size = 8192
    total_time = 0.1
 
def checkSpectrum(spec_graph):
    nyq_freq = data.sampling_frequency/2
    freq_range = np.linspace(0, nyq_freq, int(total_samples/2)+1)
    
    target_freq = 9e03
    target_loc = target_freq*len(freq_range)/nyq_freq
    start_loc = int(target_loc*0.95)
    end_loc = int(target_loc*1.05)
    spectrums = []
    for _ in range(10):
        buffer = scope.record(device_data, channel=1)
        total_samples = len(buffer)
        spectrum = np.abs(scipy.fft.rfft(buffer))*2/total_samples
        spec_graph.set_ydata(spectrum)
        plt.draw()
        plt.pause(0.01)
        spectrums.append(spectrum)
    

    return np.max(spectrum[start_loc:end_loc]), freq_range, spectrum

def checkVal(spec_graph):
    currVal, _, _ = checkSpectrum(spec_graph)
    return currVal

def safeMove(device, loc_graph, amount):
    #TODO: instead of wait_move, run a while loop periodically checking for status
    device.move_by(amount)
    while(device._is_moving()):



if __name__ == '__main__':
    try:
        device_data = device.open()
        scope.open(device_data, sampling_frequency=8.192e04, buffer_size=8192)

        # set up triggering on scope channel 1
        scope.trigger(device_data, enable=True, source=scope.trigger_source.analog, channel=1, level=0)
        
        print(Thorlabs.list_kinesis_devices())
        serialNoYaw = "27006315"
        serialNoTilt = "27006283"
        deviceYaw = Thorlabs.KinesisMotor(serialNoYaw)
        devicePitch = Thorlabs.KinesisMotor(serialNoTilt)

        time.sleep(0.5)

        print(deviceYaw.get_device_info())

        # deviceYaw.home(sync=False, force=True, timeout=5000000)
        # devicePitch.home(sync=False, force=True, timeout=5000000)
        # print(deviceYaw.is_homing())
        # print(deviceYaw.is_homed())
        # deviceYaw.wait_for_home()
        # devicePitch.wait_for_home()
        # print("done homing")

        currVal, freq_range, spectrum = checkSpectrum()
        plt.ion()
        plt.figure(0)
        spec_graph = plt.plot(freq_range, spectrum)[0]

        currPositionYaw = deviceYaw.get_position()
        currPositionPitch = devicePitch.get_position()

        deviationVal = 0.1
        binFactor = 1

        minYaw, maxYaw = findBoundaries(spec_graph, deviationVal, deviceYaw, binFactor)
        minPitch, maxPitch = findBoundaries(spec_graph, deviationVal, devicePitch, binFactor)

        amplitude_2d = getOrientationArray(spec_graph, minYaw, maxYaw, minPitch, maxPitch, deviceYaw, devicePitch)
        plt.figure(1)
        plt.imshow(amplitude_2d, cmap='hsv', interpolation='nearest')
        plt.savefig("spec_graph.png")

        deviceYaw.close()
        devicePitch.close()
        scope.close(device_data)
        device.close(device_data)
    except error as e:
        print(e)
        scope.close(device_data)
        device.close(device_data)
        deviceYaw.stop()
        devicePitch.stop()
        deviceYaw.close()
        devicePitch.close()