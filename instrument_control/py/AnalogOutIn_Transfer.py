"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2023-03-16

   Requires:                       
       Python 2.7, 3
       Device with deep analog-out and in buffer, like ADP2230, ADP3X50
"""

from ctypes import *
import time
from dwfconstants import *
import sys
import matplotlib.pyplot as plt
import numpy
import random

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("Version: "+str(version.value))

# prevent temperature drift
dwf.FDwfParamSet(DwfParamOnClose, c_int(0)) # 0 = run, 1 = stop, 2 = shutdown

cDevice = c_int()
dwf.FDwfEnum(0, byref(cDevice))
print("Number of Devices: "+str(cDevice.value))

if cDevice.value == 0:
    print("no device detected")
    quit()

hdwf = c_int(0)
for iDev in range(cDevice.value):
    devid = c_int()
    dwf.FDwfEnumDeviceType(iDev, byref(devid), 0)
    if devid.value == devidADP2230.value or devid.value == devidADP3X50.value:
        print("Opening device")
        if dwf.FDwfDeviceOpen(iDev, byref(hdwf)) == 0: 
            print("failed to open device")
            szerr = create_string_buffer(512)
            dwf.FDwfGetLastErrorMsg(szerr)
            print(szerr.value)
            quit()
        break

if hdwf.value == hdwfNone.value:
    print("No supported device found")
    quit()

# the device will only be configured when FDwf###Configure is called
dwf.FDwfDeviceAutoConfigureSet(hdwf, 0) 

hzRate = 100e6
cSamples = 1000000
rgdData = (c_double*int(cSamples))()
for i in range(cSamples):
    rgdData[i] = random.uniform(-1.0, 1.0)


print("Configure analog in")
dwf.FDwfAnalogInFrequencySet(hdwf, c_double(hzRate))
dwf.FDwfAnalogInChannelEnableSet(hdwf, 0, 1)
dwf.FDwfAnalogInChannelEnableSet(hdwf, 1, 1)
dwf.FDwfAnalogInChannelRangeSet(hdwf, -1, c_double(2.0))
dwf.FDwfAnalogInBufferSizeSet(hdwf, cSamples)
# relative to middle of the buffer, with time base/2 T0 will be the first sample
dwf.FDwfAnalogInTriggerPositionSet(hdwf, c_double(0.5*cSamples/hzRate))
dwf.FDwfAnalogInTriggerSourceSet(hdwf, trigsrcAnalogOut1) 
dwf.FDwfAnalogInConfigure(hdwf, 1, 1)

# on first connection wait for the offsets to stabilize
time.sleep(0.1)

print("Configure and start first analog out channel")
dwf.FDwfAnalogOutEnableSet(hdwf, 0, 1)
dwf.FDwfAnalogOutFunctionSet(hdwf, 0, funcPlayPattern)
dwf.FDwfAnalogOutFrequencySet(hdwf, 0, c_double(hzRate))
dwf.FDwfAnalogOutIdleSet(hdwf, 0, DwfAnalogOutIdleOffset)
dwf.FDwfAnalogOutDataSet(hdwf, 0, rgdData, c_int(cSamples))
dwf.FDwfAnalogOutRunSet(hdwf, 0, c_double(1.0*cSamples/hzRate))
dwf.FDwfAnalogOutRepeatSet(hdwf, 0, 1)
dwf.FDwfAnalogOutConfigure(hdwf, 0, 1)

sts = c_int()
while True:
    if dwf.FDwfAnalogInStatus(hdwf, 1, byref(sts)) == 0: 
        print("failed to open device")
        szerr = create_string_buffer(512)
        dwf.FDwfGetLastErrorMsg(szerr)
        print(szerr.value)
        quit()
    if sts.value == DwfStateDone.value :
        break
    time.sleep(0.01)
print("   done")

rgdData1 = (c_double*cSamples)()
rgdData2 = (c_double*cSamples)()
dwf.FDwfAnalogInStatusData(hdwf, 0, rgdData1, cSamples)
dwf.FDwfAnalogInStatusData(hdwf, 1, rgdData2, cSamples)

dwf.FDwfAnalogOutReset(hdwf, 0)
dwf.FDwfDeviceCloseAll()

plt.plot(rgdData1, color='orange', label='C1')
plt.plot(rgdData2, color='blue', label='C2')
plt.show()

