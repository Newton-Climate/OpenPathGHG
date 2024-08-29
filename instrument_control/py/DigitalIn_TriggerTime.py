"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2023-12-13

   Requires:                       
       Python 2.7, 3
"""

from ctypes import *
import math
import time
import matplotlib.pyplot as plt
import sys
import numpy
from dwfconstants import *


if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

hdwf = c_int()
sts = c_byte()

version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: "+str(version.value))

print("Opening first device")
dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

if hdwf.value == 0:
    print("failed to open device")
    szerr = create_string_buffer(512)
    dwf.FDwfGetLastErrorMsg(szerr)
    print(str(szerr.value))
    quit()


# the device will only be configured when FDwf###Configure is called
dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(0)) 

# generate 1 Hz pulse
dwf.FDwfDigitalOutEnableSet(hdwf, c_int(0), c_int(1))
dwf.FDwfDigitalOutDividerSet(hdwf, c_int(0), c_int(100000))
dwf.FDwfDigitalOutCounterSet(hdwf, c_int(0), c_int(100), c_int(900))
dwf.FDwfDigitalOutConfigure(hdwf, c_int(1))

hzDI = c_double()
dwf.FDwfDigitalInInternalClockInfo(hdwf, byref(hzDI))
print("DigitanIn base freq: "+str(hzDI.value/1e6)+"MHz")
# trigger time resolution is given by sampling rate
# for Digital Discovery at 100MHz or higher is 1.25ns
dwf.FDwfDigitalInDividerSet(hdwf, c_int(int(hzDI.value/100e6)))
dwf.FDwfDigitalInSampleFormatSet(hdwf, c_int(8))
dwf.FDwfDigitalInBufferSizeSet(hdwf, c_int(64))
dwf.FDwfDigitalInTriggerSourceSet(hdwf, trigsrcDetectorDigitalIn)
dwf.FDwfDigitalInTriggerSet(hdwf, c_int(0), c_int(0), c_int(1<<0), c_int(0)) # DIO0/DIN0 rising edge

# begin acquisition
dwf.FDwfDigitalInConfigure(hdwf, c_int(0), c_int(1))
print("Waiting for acquisition...")

print("Press Ctrl+C to stop")
try:
    while True:
        while True:
            if dwf.FDwfDigitalInStatus(hdwf, c_int(1), byref(sts)) != 1:
                szerr = create_string_buffer(512)
                dwf.FDwfGetLastErrorMsg(szerr)
                print(szerr.value)
                quit()
            if sts.value == DwfStateDone.value : # done
                break
            time.sleep(0.001)
        
        sec = c_uint()
        tick = c_uint()
        ticksec = c_uint()
        # acquisition software time for Analog Discovery and T0 with 8-10ns precision for ADP3X50
        dwf.FDwfDigitalInStatusTime(hdwf, byref(sec), byref(tick), byref(ticksec))
        s = time.localtime(sec.value)
        ns = 1e9/ticksec.value*tick.value
        ms = math.floor(ns/1e6)
        ns -= ms*1e6
        us = math.floor(ns/1e3)
        ns -= us*1e3
        ns = math.floor(ns)
        print(time.strftime("%Y-%m-%d %H:%M:%S", s)+"."+str(ms).zfill(3)+"."+str(us).zfill(3)+"."+str(ns).zfill(3))

except KeyboardInterrupt:
    pass
    
dwf.FDwfDeviceCloseAll()


