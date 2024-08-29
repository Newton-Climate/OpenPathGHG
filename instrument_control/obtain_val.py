import time
import ctypes                     # import the C compatible data types
from sys import platform, path    # this is needed to check the OS type and get the PATH
from os import sep                # OS specific file path separators
import numpy as np
import scipy
import matplotlib.pyplot as plt
from pathlib import Path
import sys
 

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

def checkVal():
    start = time.time()
    # print(scope.record(device_data, channel=1))
    buffer = scope.record(device_data, channel=1)
    total_samples = len(buffer)
    spectrum = np.abs(scipy.fft.rfft(buffer))*2/total_samples
    nyq_freq = data.sampling_frequency/2
    freq_range = np.linspace(0, nyq_freq, int(total_samples/2)+1)
    
    target_freq = 9e03
    target_loc = target_freq*len(freq_range)/nyq_freq
    start_loc = int(target_loc*0.95)
    end_loc = int(target_loc*1.05)
    end = time.time()
    print(end-start)
    graph = plt.plot(freq_range[1:], spectrum[1:])[0]
    # graph = plt.plot(buffer)[0]
    plt.draw()
    plt.pause(0.01)
    print(np.argmax(spectrum))
    print(np.max(spectrum[start_loc:end_loc]))
    print(np.max(spectrum))
    while True:
        buffer = scope.record(device_data, channel=1)
        total_samples = len(buffer)
        spectrum = np.abs(scipy.fft.rfft(buffer))*2/total_samples
        nyq_freq = data.sampling_frequency/2
        freq_range = np.linspace(0, nyq_freq, int(total_samples/2)+1)
        
        target_freq = 9e03
        target_loc = target_freq*len(freq_range)/nyq_freq
        start_loc = int(target_loc*0.95)
        end_loc = int(target_loc*1.05)
        end = time.time()
        print(buffer)
        graph.set_data(freq_range[1:], spectrum[1:])
        # graph.set_ydata(buffer)
        plt.draw()
        plt.pause(0.01)

    return np.max(spectrum[start_loc:end_loc])


try:
    device_data = device.open()
    scope.open(device_data, sampling_frequency=8.192e04, buffer_size=8192)
    scope.trigger(device_data, enable=False, source=scope.trigger_source.analog, channel=1, level=2.723)

    print(checkVal())

    # set up triggering on scope channel 1
    plt.ion()

except error as e:
    print(e)
    plt.ioff()
    scope.close(device_data)
    device.close(device_data)