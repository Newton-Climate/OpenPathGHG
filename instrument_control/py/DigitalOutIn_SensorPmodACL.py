"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2024-04-19

   Requires:                       
       Python 2.7, 3
   Description:
   
"""

from ctypes import *
from dwfconstants import *
import math
import sys
import ctypes
import time


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
dwf.FDwfDeviceOpen(-1, byref(hdwf))

if hdwf.value == 0:
    print("failed to open device")
    szerr = create_string_buffer(512)
    dwf.FDwfGetLastErrorMsg(szerr)
    print(str(szerr.value))
    quit()

dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(0)) # 0 = the device will only be configured when FDwf###Configure is called

nSamples = 100000
rgbSamples = (c_uint8*nSamples)()
cAvailable = c_int()
cLost = c_int()
cCorrupted = c_int()

idxCS = 0 # DIO-0
idxClk = 1 # DIO-1
idxMosi = 2 # DIO-2
idxMiso = 3 # DIO-3
CPOL = 1 
CPHA = 1 
hzFreq = 1e4
hzRate = 2 


# V+ 3.3V for AD2,3
dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(0), c_int(0), c_double(1)) 
dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(0), c_int(1), c_double(3.3)) 
dwf.FDwfAnalogIOEnableSet(hdwf, 1)
dwf.FDwfAnalogIOConfigure(hdwf, 1)

# Digital Discovery VIO 3.3V and enable output
# dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(0), c_int(0), c_double(3.3)) 
# dwf.FDwfAnalogIOEnableSet(hdwf, 1)
# dwf.FDwfAnalogIOConfigure(hdwf, 1)

time.sleep(0.2) 

# for Digital Discovery 
# with order 0: DIN0:7;   with 32 bit sampling [DIN0:23  + DIO24:31]
# with order 1: DIO24:31; with 32 bit sampling [DIO24:39 + DIN0:15]
dwf.FDwfDigitalInInputOrderSet(hdwf, 1)

hzSys = c_double()
dwf.FDwfDigitalOutInternalClockInfo(hdwf, byref(hzSys))

# Select 
dwf.FDwfDigitalOutEnableSet(hdwf, idxCS, 1)
# output high while DigitalOut not running
dwf.FDwfDigitalOutIdleSet(hdwf, idxCS, DwfDigitalOutIdleHigh) 
# output constant low while running
dwf.FDwfDigitalOutCounterInitSet(hdwf, idxCS, 0, 0)
dwf.FDwfDigitalOutCounterSet(hdwf, idxCS, 0, 0)

# Clock
dwf.FDwfDigitalOutEnableSet(hdwf, idxClk, 1)
# set prescaler twice of SPI frequency
dwf.FDwfDigitalOutDividerSet(hdwf, idxClk, int(hzSys.value/hzFreq/2))
# 1 tick low, 1 tick high
dwf.FDwfDigitalOutCounterSet(hdwf, idxClk, 1, 1)
# start with low or high based on clock polarity
dwf.FDwfDigitalOutCounterInitSet(hdwf, idxClk, CPOL, 1)
dwf.FDwfDigitalOutIdleSet(hdwf, idxClk, 1+CPOL) # 1=DwfDigitalOutIdleLow 2=DwfDigitalOutIdleHigh

# MOSI
dwf.FDwfDigitalOutEnableSet(hdwf, idxMosi, 1)
dwf.FDwfDigitalOutTypeSet(hdwf, idxMosi, DwfDigitalOutTypeCustom) 
# for high active clock, hold the first bit for 1.5 periods 
dwf.FDwfDigitalOutDividerInitSet(hdwf, idxMosi, int((1+0.5*CPHA)*hzSys.value/hzFreq)) 
# SPI frequency, bit frequency
dwf.FDwfDigitalOutDividerSet(hdwf, idxMosi, int(hzSys.value/hzFreq))

dwf.FDwfDigitalOutConfigure(hdwf, 0)
time.sleep(0.1) 

def reversebits(word, bits=8): # swap bit order, MSB/LSB
    result = 0
    for i in range(bits):
        result <<= 1
        result |= word & 1
        word >>= 1
    return result

def send(data):
    rgdData = (len(data)*c_byte)()
    for i in range(len(data)):
        rgdData[i] = reversebits(data[i])
    # data sent out LSB first
    dwf.FDwfDigitalOutDataSet(hdwf, idxMosi, byref(rgdData), 8*len(data))
    dwf.FDwfDigitalOutRunSet(hdwf, c_double((8*len(data)+0.5)/hzFreq))
    dwf.FDwfDigitalOutConfigure(hdwf, 1)
    while True:
        if dwf.FDwfDigitalOutStatus(hdwf, byref(sts)) == 0:
            print("Error:")
            szerr = create_string_buffer(512)
            dwf.FDwfGetLastErrorMsg(szerr)
            print(str(szerr.value))
            quit()
        if sts.value == DwfStateDone.value:
            break
            

send([0x2D, 0x08]) # POWER_CTL | Measure
send([0x31, 0x08]) # DATA_FORMAT | FULL_RES


print("Configuring SPI spy...")
# record mode
dwf.FDwfDigitalInAcquisitionModeSet(hdwf, acqmodeRecord)
# for sync mode set divider to -1 
dwf.FDwfDigitalInDividerSet(hdwf, -1)
# 8bit per sample format DIO 0:7
dwf.FDwfDigitalInSampleFormatSet(hdwf, 8)
# continuous sampling 
dwf.FDwfDigitalInTriggerPositionSet(hdwf, -1)
# in sync mode the trigger is used for sampling condition
# trigger detector mask:       low&high&(rising    | falling)
dwf.FDwfDigitalInTriggerSet(hdwf, 0, 0, c_int(1<<idxClk), 0)
# sample on clock rising edge for sampling bits, or CS rising edge to detect frames
dwf.FDwfDigitalInConfigure(hdwf, 1, 1)

# send 0xF2 command byte and read 3*2 bytes, 7 bytes in total
rgbData = (7*c_byte)(reversebits(0xF2),0,0,0,0,0,0)
dwf.FDwfDigitalOutDataSet(hdwf, idxMosi, byref(rgbData), 7*8)
sRun = (7*8+0.5)/hzFreq
dwf.FDwfDigitalOutRunSet(hdwf, c_double(sRun))
dwf.FDwfDigitalOutWaitSet(hdwf, c_double(1.0/hzRate-sRun))
dwf.FDwfDigitalOutRepeatSet(hdwf, 0)
dwf.FDwfDigitalOutConfigure(hdwf, 1)

try:
    cSamples = 0
    while True:
        if dwf.FDwfDigitalInStatus(hdwf, 1, byref(sts)) == 0:
            print("Error:")
            szerr = create_string_buffer(512)
            dwf.FDwfGetLastErrorMsg(szerr)
            print(str(szerr.value))
            quit()
            
        dwf.FDwfDigitalInStatusRecord(hdwf, byref(cAvailable), byref(cLost), byref(cCorrupted))
        if cLost.value :
            print("Samples were lost!")
        if cCorrupted.value :
            print("Samples could be corrupted!")
        if cSamples+cAvailable.value > nSamples :
            cAvailable = c_int(nSamples)

        dwf.FDwfDigitalInStatusData(hdwf, byref(rgbSamples, cSamples), cAvailable) # 8bit data
        cSamples += cAvailable.value
        
        while cSamples > 7*8:
            cSamples -= 7*8
            XYZ = []
            iBit = 0
            for j in range(3):
                iBit += 8 # skip command
                fsMiso = 0
                for i in range(16):
                    fsMiso <<= 1
                    if(1&(rgbSamples[iBit]>>idxMiso)):
                        fsMiso |= 1
                    iBit += 1
                XYZ.append(fsMiso)
                #XYZ.append(1.0*fsMiso/256/256/256)
            print("XYZ: %d %d %d" % (XYZ[0], XYZ[1], XYZ[2]))
            for i in range(cSamples): # shift remaining samples
                rgbSamples[i] = rgbSamples[i+7*8]
except KeyboardInterrupt:
    pass

dwf.FDwfDeviceClose(hdwf)
