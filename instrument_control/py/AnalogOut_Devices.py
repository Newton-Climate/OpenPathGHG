"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2024-06-21

   Requires:                       
       Python 2.7, 3
"""

from ctypes import *
import time
from dwfconstants import *
import sys

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

hdwf = c_int()
channel = c_int(0)

version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: "+str(version.value))

# prevent temperature drift
dwf.FDwfParamSet(DwfParamOnClose, c_int(0)) # 0 = run, 1 = stop, 2 = shutdown
dwf.FDwfParamSet(DwfParamExtFreq, 10000000) # reference clock frequency
dwf.FDwfParamSet(DwfParamFrequency, 100000000) # system clock frequency

cDevice = c_int()
dwf.FDwfEnum(0, byref(cDevice))
print(f"Found {cDevice.value} devices")
print("Connect trigger lines between devices for reference clock and triggering\nTwist each wire with ground for shielding")

hzFreq = 1000
cSteps = 256

hdwf = c_int(0)
hdwfMaster = c_int(0)

for iDevice in range(cDevice.value):
    dwf.FDwfDeviceOpen(iDevice, byref(hdwf))
    if hdwf.value == 0:
        continue
    
    dwf.FDwfDeviceAutoConfigureSet(hdwf, 0) # the instruments will only be configured when FDwf###Configure is called

    if hdwfMaster.value == 0:
        hdwfMaster.value = hdwf.value
        dwf.FDwfDeviceParamSet(hdwf, DwfParamClockMode, 1) # reference clock output on Trigger 1
        dwf.FDwfDeviceTriggerSet(hdwf, 1, trigsrcPC) # Trigger 2 outputs TriggerPC
    else:
        dwf.FDwfDeviceParamSet(hdwf, DwfParamClockMode, 2) # reference clock input on Trigger 1
        #dwf.FDwfDeviceParamSet(hdwf, DwfParamFreqPhase, 180) # phase for slave device, if needed
        
    
    cChannel = c_int()
    dwf.FDwfAnalogOutCount(hdwf, byref(cChannel))
    if cChannel.value > 2: cChannel.value = 2
    
    for iChannel in range(cChannel.value):
        print(f"Configure Device {iDevice +1} W{iChannel+1}")
        dwf.FDwfAnalogOutTriggerSourceSet(hdwf, iChannel, trigsrcExternal2)
        #dwf.FDwfAnalogOutRunSet(hdwf, iChannel, c_double(cSteps/hzFreq)) # run length
        #dwf.FDwfAnalogOutRepeatSet(hdwf, iChannel, 1) # run once
        
        if hdwfMaster.value == hdwf.value :
            dwf.FDwfAnalogOutWaitSet(hdwf, iChannel, c_double(1e-8)) # if needed
        
        dwf.FDwfAnalogOutNodeEnableSet(hdwf, iChannel, AnalogOutNodeCarrier, c_int(1))
        dwf.FDwfAnalogOutNodeFunctionSet(hdwf, iChannel, AnalogOutNodeCarrier, funcPulse)
        dwf.FDwfAnalogOutNodeFrequencySet(hdwf, iChannel, AnalogOutNodeCarrier, c_double(hzFreq))
        dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, iChannel, AnalogOutNodeCarrier, c_double(1.0))
        
        if hdwfMaster.value == hdwf.value and iChannel == 0:
            rgCustom = (c_double*cSteps)()
            for i in range(cSteps):
                rgCustom[i] = 1-2*i/cSteps
                
            dwf.FDwfAnalogOutNodeEnableSet(hdwf, iChannel, AnalogOutNodeAM, c_int(1))
            dwf.FDwfAnalogOutNodeFunctionSet(hdwf, iChannel, AnalogOutNodeAM, funcCustom)
            dwf.FDwfAnalogOutNodeDataSet(hdwf, iChannel, AnalogOutNodeAM, rgCustom, cSteps)
            dwf.FDwfAnalogOutNodeFrequencySet(hdwf, iChannel, AnalogOutNodeAM, c_double(hzFreq/cSteps))
            dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, iChannel, AnalogOutNodeAM, c_double(100.0))
            dwf.FDwfAnalogOutNodeOffsetSet(hdwf, iChannel, AnalogOutNodeAM, c_double(-100.0))
        
        dwf.FDwfAnalogOutConfigure(hdwf, iChannel, 1)

print("Triggering to start generators...")
dwf.FDwfDeviceTriggerPC(hdwfMaster)
dwf.FDwfDeviceCloseAll()
