"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2024-07-26

   Requires:                       
       Python 2.7, 3
"""

from ctypes import *
import time
from dwfconstants import *
import sys
import numpy
import matplotlib.pyplot as plt


if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

channel = c_int(0)

version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: "+str(version.value))

# prevent temperature drift
dwf.FDwfParamSet(DwfParamOnClose, 0) # 0 = run, 1 = stop, 2 = shutdown
dwf.FDwfParamSet(DwfParamExtFreq, 10000000) # reference clock frequency
dwf.FDwfParamSet(DwfParamFrequency, 100000000) # system clock frequency

cDevice = c_int(0)
dwf.FDwfEnum(0, byref(cDevice))
print(f"Found {cDevice.value} devices")
print("Connect trigger lines between devices for reference clock and triggering\nUse coax cable or twist each wire with ground for shielding")

rghdwf = []
cSamples = 1024
rgdSamples = (c_double*cSamples)()
hzRate = 1e8

for iDevice in range(cDevice.value):
    hdwf = c_int(0)
    dwf.FDwfDeviceOpen(iDevice, byref(hdwf))
    if hdwf.value == 0:
        continue
    
    dwf.FDwfDeviceAutoConfigureSet(hdwf, 0) # the instruments will only be configured when FDwf###Configure is called
    rghdwf.append(hdwf.value)
    
    if len(rghdwf) == 1: # let use first device as master
        dwf.FDwfDeviceParamSet(hdwf, DwfParamClockMode, 1) # reference clock output on Trigger 1
        dwf.FDwfDeviceTriggerSet(hdwf, 1, trigsrcAnalogIn) # master Trigger 2 output
        # master with default trigsrcNone will trigger immediately
        dwf.FDwfAnalogInTriggerPositionSet(hdwf, c_double(0.0)) 
    else:
        dwf.FDwfDeviceParamSet(hdwf, DwfParamClockMode, 2) # reference clock input on Trigger 1
        #dwf.FDwfDeviceParamSet(hdwf, DwfParamFreqPhase, 180) # phase for slave device, if needed
        dwf.FDwfAnalogInTriggerSourceSet(hdwf, trigsrcExternal2) 
        dwf.FDwfAnalogInTriggerPositionSet(hdwf, c_double(0.0 - 10/hzRate)) # compensate trigger delay to slave
        
    print(f"Configure Device {iDevice+1}")
    dwf.FDwfAnalogInFrequencySet(hdwf, c_double(hzRate))
    dwf.FDwfAnalogInBufferSizeSet(hdwf, cSamples) 
    dwf.FDwfAnalogInChannelEnableSet(hdwf, -1, 1)
    dwf.FDwfAnalogInChannelRangeSet(hdwf, -1, c_double(2.0))
    dwf.FDwfAnalogInConfigure(hdwf, 1, 0)

if len(rghdwf)<1:
    print("No device detected")
    exit()

# wait on first open, offset or range adjustment the levels to settle
time.sleep(0.1)

# start oscilloscopes
for iDevice in range(len(rghdwf)):
    if iDevice == 0: continue # master/first device armed last
    dwf.FDwfAnalogInConfigure(rghdwf[iDevice], 1, 1)
dwf.FDwfAnalogInConfigure(rghdwf[0], 1, 1)

# wait for master to finish the capture
while True:
    sts = c_byte(0)
    dwf.FDwfAnalogInStatus(rghdwf[0], 1, byref(sts))
    if sts.value == DwfStateDone.value :
        break

for iDevice in range(len(rghdwf)):
    sts = c_byte(0)
    if iDevice == 0: continue # master/first device armed last
    dwf.FDwfAnalogInStatus(rghdwf[iDevice], 1, byref(sts))
    if sts.value != DwfStateDone.value :
        print(f"Device {iDevice+1} not done")
        exit()
        break
        
# get data
for iDevice in range(len(rghdwf)):
    cChannel = c_int(0)
    dwf.FDwfAnalogInChannelCount(rghdwf[iDevice], byref(cChannel))
    for iChannel in range(cChannel.value):
        dwf.FDwfAnalogInStatusData(rghdwf[iDevice], iChannel, rgdSamples, cSamples)
        plt.plot(numpy.fromiter(rgdSamples, dtype = numpy.float), label=f'D{iDevice+1}C{iChannel+1}')
    
dwf.FDwfDeviceCloseAll()

plt.show()
