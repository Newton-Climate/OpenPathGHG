"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2020-12-09

   Requires:                       
       Python 2.7, 3
   NPN transistor test
"""

from ctypes import *
import time
from dwfconstants import *
import sys
import matplotlib.pyplot as plt
import numpy

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("Version: "+str(version.value))

cdevices = c_int()
dwf.FDwfEnum(0, byref(cdevices))
print("Number of Devices: "+str(cdevices.value))

if cdevices.value == 0:
    print("no device detected")
    quit()

dwf.FDwfParamSet(DwfParamOnClose, 0) # 0 = run, 1 = stop, 2 = shutdown

print("Opening first device")
hdwf = c_int()
dwf.FDwfDeviceOpen(-1, byref(hdwf))

if hdwf.value == hdwfNone.value:
    print("failed to open device")
    quit()

dwf.FDwfDeviceAutoConfigureSet(hdwf, 0) # 0 = the device will only be configured when FDwf###Configure is called

print("Configuring device...")
# collector: 0V to 5V triangle output and 50Hz, 20ms
dwf.FDwfAnalogOutEnableSet(hdwf, 0, 1) 
dwf.FDwfAnalogOutFunctionSet(hdwf, 0, funcTriangle)
dwf.FDwfAnalogOutPhaseSet(hdwf, 0, c_double(270.0))
dwf.FDwfAnalogOutFrequencySet(hdwf, 0, c_double(50))
dwf.FDwfAnalogOutOffsetSet(hdwf, 0, c_double(2.5))
dwf.FDwfAnalogOutAmplitudeSet(hdwf, 0, c_double(2.5))
dwf.FDwfAnalogOutMasterSet(hdwf, 0, 1);
dwf.FDwfAnalogOutConfigure(hdwf, 0, 0)

# base: 1V to 2V in 5 steps at 10Hz, 100ms total length
dwf.FDwfAnalogOutEnableSet(hdwf, 1, 1) 
dwf.FDwfAnalogOutFunctionSet(hdwf, 1, funcCustom)
dwf.FDwfAnalogOutFrequencySet(hdwf, 1, c_double(10))
dwf.FDwfAnalogOutOffsetSet(hdwf, 1, c_double(1.5))
dwf.FDwfAnalogOutAmplitudeSet(hdwf, 1, c_double(0.5))
# values normalized to +-1 
rgSteps = (c_double*5)(-1.0, -0.5, 0, 0.5, 1.0)
dwf.FDwfAnalogOutDataSet(hdwf, 1, rgSteps, len(rgSteps))
dwf.FDwfAnalogOutRunSet(hdwf, 1, c_double(0.1))
dwf.FDwfAnalogOutRepeatSet(hdwf, 1, 1)
dwf.FDwfAnalogOutConfigure(hdwf, 1, 0)

# scope: 5000 samples at 50kHz, 100ms
dwf.FDwfAnalogInChannelEnableSet(hdwf, 0, 1)
dwf.FDwfAnalogInChannelEnableSet(hdwf, 1, 1)
dwf.FDwfAnalogInFrequencySet(hdwf, c_double(50e3))
dwf.FDwfAnalogInChannelRangeSet(hdwf, 0, c_double(10.0))
dwf.FDwfAnalogInChannelRangeSet(hdwf, 1, c_double(10.0))
dwf.FDwfAnalogInBufferSizeSet(hdwf, 5000)
dwf.FDwfAnalogInTriggerSourceSet(hdwf, trigsrcAnalogOut2) 
dwf.FDwfAnalogInTriggerPositionSet(hdwf, c_double(0.05)) # 5ms, trigger at first sample
dwf.FDwfAnalogInConfigure(hdwf, 1, 0)

print("Wait for the offset to stabilize...")
time.sleep(1)

print("Starting test...")
dwf.FDwfAnalogInConfigure(hdwf, 0, 1)
dwf.FDwfAnalogOutConfigure(hdwf, 1, 1)

sts = c_int()
while True:
    dwf.FDwfAnalogInStatus(hdwf, 1, byref(sts))
    if sts.value == DwfStateDone.value :
        break
    time.sleep(0.001)
print("done")

rgc1 = (c_double*5000)()
rgc2 = (c_double*5000)()
dwf.FDwfAnalogInStatusData(hdwf, 0, rgc1, len(rgc1)) # get channel 1 data
dwf.FDwfAnalogInStatusData(hdwf, 1, rgc2, len(rgc2)) # get channel 2 data

dwf.FDwfDeviceCloseAll()

plt.plot(numpy.fromiter(rgc2, dtype = numpy.float), numpy.fromiter(rgc1, dtype = numpy.float))
plt.show()

