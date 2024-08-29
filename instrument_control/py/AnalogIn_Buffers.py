"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2023-12-29

   Requires:                       
       Python 2.7, 3
"""

from ctypes import *
from dwfconstants import *
import math
import time
import sys
import numpy


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

dwf.FDwfParamSet(DwfParamOnClose, c_int(0)) # 0 = run, 1 = stop, 2 = shutdown

print("Opening first device")
dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))
# 2nd configuration for Analog Discovery with 16k analog-in buffer
#dwf.FDwfDeviceConfigOpen(c_int(-1), c_int(1), byref(hdwf)) 

if hdwf.value == hdwfNone.value:
    szError = create_string_buffer(512)
    dwf.FDwfGetLastErrorMsg(szError);
    print("failed to open device\n"+str(szError.value))
    quit()

dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(0)) # 0 = the device will only be configured when FDwf###Configure is called

print("Generating signal...")
#                                    AWG 1     carrier
dwf.FDwfAnalogOutNodeEnableSet(hdwf, c_int(0), c_int(0), c_int(1))
dwf.FDwfAnalogOutNodeFunctionSet(hdwf, c_int(0), c_int(0), funcSquare)
dwf.FDwfAnalogOutNodeFrequencySet(hdwf, c_int(0), c_int(0), c_double(10e6))
dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, c_int(0), c_int(0), c_double(1.0))
dwf.FDwfAnalogOutConfigure(hdwf, c_int(0), c_int(1))

cSamples = c_int(8192)
hzRate = c_double(100e6)
cBuffers = c_int(1000) # up to 32768 or device buffer size (ADP3x50 128Mi/capture size)
rgdSamples = (c_double*cSamples.value)()

#set up acquisition
dwf.FDwfAnalogInFrequencySet(hdwf, hzRate)
dwf.FDwfAnalogInBufferSizeSet(hdwf, cSamples) 
dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(0), c_int(1))
dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(0), c_double(5))
dwf.FDwfAnalogInBuffersSet(hdwf, cBuffers)

#set up trigger
dwf.FDwfAnalogInTriggerAutoTimeoutSet(hdwf, c_double(0)) #disable auto trigger
dwf.FDwfAnalogInTriggerSourceSet(hdwf, trigsrcDetectorAnalogIn) #one of the analog in channels
dwf.FDwfAnalogInTriggerTypeSet(hdwf, trigtypeEdge)
dwf.FDwfAnalogInTriggerChannelSet(hdwf, c_int(0)) # first channel
dwf.FDwfAnalogInTriggerLevelSet(hdwf, c_double(0.0)) # 0.0V
dwf.FDwfAnalogInTriggerConditionSet(hdwf, DwfTriggerSlopeRise) 
# relative to middle of the buffer, with time base/2 T0 will bt the first sample
dwf.FDwfAnalogInTriggerPositionSet(hdwf, c_double(0.5*cSamples.value/hzRate.value)) 

dwf.FDwfAnalogInConfigure(hdwf, c_int(1), c_int(0))

dwf.FDwfAnalogInBuffersGet(hdwf, byref(cBuffers))
dwf.FDwfAnalogInFrequencyGet(hdwf, byref(hzRate))
dwf.FDwfAnalogInBufferSizeGet(hdwf, byref(cSamples))

print(f"Device Buffers: {cBuffers.value}")
print(f"Capture: {cSamples.value} samples at {hzRate.value/1e6} MHz {cSamples.value/hzRate.value*1e6} us")

# wait for the offset to stabilize, before the first reading after device open or offset/range change
time.sleep(1)

print("Starting repeated acquisitions...")
dwf.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))

dtTrigger = []
psec = c_uint()
ptick = c_uint()
ticksec = c_uint()

tStart = time.time()
for iCapture in range(cBuffers.value):
    # new acquisition is started automatically after done state 
    while True:
        if dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts)) != 1:
            szerr = create_string_buffer(512)
            dwf.FDwfGetLastErrorMsg(szerr)
            print(szerr.value)
            quit()
        if sts.value == DwfStateDone.value :
            break
    
    #dwf.FDwfAnalogInStatusData(hdwf, 0, rgdSamples, cSamples) # get channel 1 data
    sec = c_uint()
    tick = c_uint()

    # hardware trigger time for newer devices AD3, ADP3X50... and software time for older devices
    dwf.FDwfAnalogInStatusTime(hdwf, byref(sec), byref(tick), byref(ticksec))

    if iCapture == 0:
        s = time.localtime(sec.value)
        ns = 1e9/ticksec.value*tick.value
        ms = math.floor(ns/1e6)
        ns -= ms*1e6
        us = math.floor(ns/1e3)
        ns -= us*1e3
        ns = math.floor(ns)
        print("First T0: "+time.strftime("%Y-%m-%d %H:%M:%S", s)+"."+str(ms).zfill(3)+"."+str(us).zfill(3)+"."+str(ns).zfill(3))
    else:
        dtTrigger.append((sec.value-psec.value)*ticksec.value + (tick.value-ptick.value))
    psec = sec
    ptick = tick
    
tEnd = time.time()

print(f"Hadrware dT0 min/max {min(dtTrigger)*1e6/ticksec.value}/{max(dtTrigger)*1e6/ticksec.value} us")
print(f"Capture gap min/max {round((min(dtTrigger)/ticksec.value-cSamples.value/hzRate.value)*1e6,3)}/{round((max(dtTrigger)/ticksec.value-cSamples.value/hzRate.value)*1e6,3)} us")
print(f"Software elapsed: {round((tEnd-tStart)*1e3,3)} ms")
print(f"Average capture transfer time: {round((tEnd-tStart)/cBuffers.value*1e6,3)} us")

dwf.FDwfAnalogOutConfigure(hdwf, c_int(0), c_int(0))
dwf.FDwfDeviceCloseAll()



