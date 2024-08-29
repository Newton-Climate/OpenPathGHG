"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision: 13/6/2024
   Requires:
"""

from ctypes import *
from dwfconstants import *
import math
import time
import sys

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print(f"DWF Version: {version.value}")

dwf.FDwfParamSet(DwfParamOnClose, 0) # continue running after device close

cDevice = c_int()

dwf.FDwfEnum(0, byref(cDevice))
print(f"Found {cDevice.value} devices")

print("Connect trigger lines between devices for reference clock and triggering")

hdwf = c_int(0)
hdwfMaster = c_int(0)

dwf.FDwfParamSet(DwfParamExtFreq, 10000000) # reference clock frequency
dwf.FDwfParamSet(DwfParamFrequency, 100000000) # system clock frequency

# Open device
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
        
    print(f"Configure Device {iDevice +1}")
    
    dwf.FDwfDigitalOutTriggerSourceSet(hdwf, trigsrcExternal2)
    hzSys = c_double()
    dwf.FDwfDigitalOutInternalClockInfo(hdwf, byref(hzSys))
    cChannel = c_int()
    dwf.FDwfDigitalOutCount(hdwf, byref(cChannel))

    for iChannel in range(cChannel.value):
        dwf.FDwfDigitalOutEnableSet(hdwf, iChannel, 1)  # DIO-0
        dwf.FDwfDigitalOutDividerSet(hdwf, iChannel, int(hzSys.value / 1e4 / 2))  # prescaler to 2kHz, SystemFrequency/1kHz/2
        dwf.FDwfDigitalOutCounterSet(hdwf, iChannel, 1, 1)  # 1 tick low, 1 tick high
        
    dwf.FDwfDigitalOutConfigure(hdwf, 1)

print("Triggering to start generators...")
dwf.FDwfDeviceTriggerPC(hdwfMaster)
dwf.FDwfDeviceCloseAll()
