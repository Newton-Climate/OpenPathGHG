"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2024-04-10

   Requires:                       
       Python 2.7, 3
"""

from ctypes import *
from dwfconstants import *
import math
import time
import sys
import numpy
# import matplotlib.pyplot as plt



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

# prevent temperature drift
dwf.FDwfParamSet(DwfParamOnClose, c_int(0)) # 0 = run, 1 = stop, 2 = shutdown

print("Opening first device")
dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

if hdwf.value == hdwfNone.value:
    szError = create_string_buffer(512)
    dwf.FDwfGetLastErrorMsg(szError);
    print("failed to open device\n"+str(szError.value))
    quit()

dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(0)) # 0 = the device will only be configured when FDwf###Configure is called

print("Generating signal...")
#                                    AWG 1     carrier
dwf.FDwfAnalogOutNodeEnableSet(hdwf, c_int(0), c_int(0), c_int(1))
dwf.FDwfAnalogOutNodeFunctionSet(hdwf, c_int(0), c_int(0), funcSine)
dwf.FDwfAnalogOutNodeFrequencySet(hdwf, c_int(0), c_int(0), c_double(1e3))
dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, c_int(0), c_int(0), c_double(1.0))
dwf.FDwfAnalogOutConfigure(hdwf, c_int(0), c_int(1))

cCaptures = 102
cSamples = 4096
hzRate = 100e6

#set up acquisition
dwf.FDwfAnalogInFrequencySet(hdwf, c_double(hzRate))
dwf.FDwfAnalogInBufferSizeSet(hdwf, c_int(cSamples)) 
dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(0), c_int(1))
dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(1), c_int(1))
dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(0), c_double(5))
dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(1), c_double(5))

#set up trigger
dwf.FDwfAnalogInTriggerAutoTimeoutSet(hdwf, c_double(0)) #disable auto trigger
dwf.FDwfAnalogInTriggerSourceSet(hdwf, trigsrcDetectorAnalogIn) #one of the analog in channels
dwf.FDwfAnalogInTriggerTypeSet(hdwf, trigtypeEdge)
dwf.FDwfAnalogInTriggerChannelSet(hdwf, c_int(0)) # first channel
dwf.FDwfAnalogInTriggerLevelSet(hdwf, c_double(0.0)) # 0.0V
dwf.FDwfAnalogInTriggerHysteresisSet(hdwf, c_double(0.01)) # 10mV
dwf.FDwfAnalogInTriggerConditionSet(hdwf, DwfTriggerSlopeRise) 
# relative to middle of the buffer, with time base/2 T0 will be the first sample
dwf.FDwfAnalogInTriggerPositionSet(hdwf, c_double(0.5*cSamples/hzRate)) 
dwf.FDwfAnalogInConfigure(hdwf, c_int(1), c_int(0))
# wait for the offset to stabilize, before the first reading after device open or offset/range change
time.sleep(1)


sec = c_uint()
tick = c_uint()
ticksec = c_uint()
datetime = []
channel1 = []
channel2 = []
for i in range(cCaptures):
    channel1.append((c_double*cSamples)())
    channel2.append((c_double*cSamples)())

print("Starting repeated acquisitions")
dwf.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))

for i in range(cCaptures):
    # new acquisition is started automatically after done state 

    while True:
        if dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts)) != 1:
            szError = create_string_buffer(512)
            dwf.FDwfGetLastErrorMsg(szError);
            print("failed to open device\n"+str(szError.value))
            quit()
        if sts.value == DwfStateDone.value :
            break
    
    dwf.FDwfAnalogInStatusData(hdwf, 0, channel1[i], cSamples) # get channel 1 data
    dwf.FDwfAnalogInStatusData(hdwf, 1, channel2[i], cSamples) # get channel 2 data
   
    dwf.FDwfAnalogInStatusTime(hdwf, byref(sec), byref(tick), byref(ticksec))
    datetime.append([sec.value,tick.value,ticksec.value])

def strdatetime(dt):
    s = time.localtime(dt[0])
    ns = 1e9*dt[1]/dt[2]
    ms = math.floor(ns/1e6)
    ns -= ms*1e6
    us = math.floor(ns/1e3)
    ns -= us*1e3
    ns = math.floor(ns)
    return time.strftime("%Y-%m-%d %H:%M:%S", s)+"."+str(ms).zfill(3)+"."+str(us).zfill(3)+"."+str(ns).zfill(3)


print("captures "+str(cCaptures))
print("first: "+strdatetime(datetime[0]))
print("2nd: "+strdatetime(datetime[1]))
print("3rd: "+strdatetime(datetime[2]))
print("4th: "+strdatetime(datetime[3]))
print("5th: "+strdatetime(datetime[4]))
print("6th: "+strdatetime(datetime[5]))
print("last: "+strdatetime(datetime[cCaptures-1]))


dwf.FDwfAnalogOutConfigure(hdwf, c_int(0), c_int(0))
dwf.FDwfDeviceCloseAll()

# plt.plot(numpy.fromiter(channel1[0], dtype = numpy.float))
# plt.plot(numpy.fromiter(channel2[0], dtype = numpy.float))
# plt.plot(numpy.fromiter(channel1[1], dtype = numpy.float))
# plt.plot(numpy.fromiter(channel2[1], dtype = numpy.float))
# plt.show()

