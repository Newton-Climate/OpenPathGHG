"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2023-11-23

   Requires:                       
       Python 2.7, 3
   Description:
   Play mono WAV file on AWG1 channel 1
   Record to stereo WAV file from Scope 1 and 2
"""

from dwfconstants import *
import ctypes
import sys
import wave
import numpy
import math
import matplotlib.pyplot as plt


rate = 100e3
length = 60000
data = (c_double*int(length))()
for i in range(length):
    data[i] = math.sin(math.pi*2*i/length)
sRun = 1.0*length/rate


if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: "+str(version.value))

dwf.FDwfParamSet(DwfParamOnClose, c_int(0)) # 0 = run, 1 = stop, 2 = shutdown

hdwf = c_int()
print("Opening first device...")
dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

if hdwf.value == 0:
    print("Failed to open device")
    szerr = create_string_buffer(512)
    dwf.FDwfGetLastErrorMsg(szerr)
    print(str(szerr.value))
    quit()

# the device will only be configured when FDwf###Configure is called
dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(0)) 

print("Staring record...")
iRecord = 0
record1 = (c_double*length)()
record2 = (c_double*length)()
#set up acquisition
dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(0), c_int(1)) # channel 1
dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(1), c_int(1)) # channel 2
dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(-1), c_double(5.0))
dwf.FDwfAnalogInChannelOffsetSet(hdwf, c_int(-1), c_double(0))
dwf.FDwfAnalogInAcquisitionModeSet(hdwf, acqmodeRecord)
dwf.FDwfAnalogInFrequencySet(hdwf, c_double(rate))
dwf.FDwfAnalogInRecordLengthSet(hdwf, c_double(sRun))
dwf.FDwfAnalogInTriggerPositionSet(hdwf, c_double(0))
dwf.FDwfAnalogInTriggerSourceSet(hdwf, trigsrcAnalogOut1)
dwf.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))


print("Playing audio...")
iPlay = 0
channel = c_int(0) # AWG 1
dwf.FDwfAnalogOutNodeEnableSet(hdwf, channel, 0, c_int(1))
dwf.FDwfAnalogOutNodeFunctionSet(hdwf, channel, 0, funcPlay)
dwf.FDwfAnalogOutRepeatSet(hdwf, channel, c_int(1))
print("Length: "+str(sRun))
dwf.FDwfAnalogOutRunSet(hdwf, channel, c_double(sRun))
dwf.FDwfAnalogOutNodeFrequencySet(hdwf, channel, 0, c_double(rate))
dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, channel, 0, c_double(2.0))
# prime the buffer with the first chunk of data
cBuffer = c_int(0)
dwf.FDwfAnalogOutNodeDataInfo(hdwf, channel, 0, 0, byref(cBuffer))
if cBuffer.value > length : cBuffer.value = length
dwf.FDwfAnalogOutNodeDataSet(hdwf, channel, 0, data, cBuffer)
iPlay += cBuffer.value
dwf.FDwfAnalogOutConfigure(hdwf, channel, c_int(1))


dataLost = c_int(0)
dataFree = c_int(0)
dataAvailable = c_int(0)
dataCorrupted = c_int(0)
sts = c_ubyte(0)
totalLost = 0
totalCorrupted = 0

# loop to send out and read in data chunks
while iRecord < length :
    if dwf.FDwfAnalogOutStatus(hdwf, channel, byref(sts)) != 1: # handle error
        print("Error")
        szerr = create_string_buffer(512)
        dwf.FDwfGetLastErrorMsg(szerr)
        print(szerr.value)
        break
    
    # play, analog out data chunk
    if sts.value == DwfStateRunning.value and iPlay < length :  # running and more data to stream
        dwf.FDwfAnalogOutNodePlayStatus(hdwf, channel, 0, byref(dataFree), byref(dataLost), byref(dataCorrupted))
        totalLost += dataLost.value
        totalCorrupted += dataCorrupted.value
        if iPlay + dataFree.value > length : # last chunk might be less than the free buffer size
            dataFree.value = length - iPlay
        if dataFree.value > 0 : 
            if dwf.FDwfAnalogOutNodePlayData(hdwf, channel, 0, byref(data, iPlay*sizeof(c_double)), dataFree) != 1: # offset for double is *8 (bytes) 
                print("Error")
                break
            iPlay += dataFree.value
    
    if dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts)) != 1: # handle error
        print("Error")
        szerr = create_string_buffer(512)
        dwf.FDwfGetLastErrorMsg(szerr)
        print(szerr.value)
        break
    
    # record, analog in data chunk
    if sts.value == DwfStateRunning.value or sts.value == DwfStateDone.value : # recording or done
        dwf.FDwfAnalogInStatusRecord(hdwf, byref(dataAvailable), byref(dataLost), byref(dataCorrupted))
        iRecord += dataLost.value
        totalLost += dataLost.value
        totalCorrupted += dataCorrupted.value
        if dataAvailable.value > 0 :
            if iRecord+dataAvailable.value > length :
                dataAvailable = c_int(length-iRecord)
            dwf.FDwfAnalogInStatusData(hdwf, c_int(0), byref(record1, sizeof(c_double)*iRecord), dataAvailable) # get channel 1 data chunk
            dwf.FDwfAnalogInStatusData(hdwf, c_int(1), byref(record2, sizeof(c_double)*iRecord), dataAvailable) # get channel 2 data chunk
            iRecord += dataAvailable.value


print("Lost: "+str(totalLost))
print("Corrupted: "+str(totalCorrupted))
print("done")
dwf.FDwfAnalogOutReset(hdwf, channel)
dwf.FDwfDeviceClose(hdwf)

plt.plot(numpy.fromiter(record1, dtype = numpy.float))
plt.plot(numpy.fromiter(record2, dtype = numpy.float))
plt.show()


