"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2023-02-29

   Requires:                       
       Python 2.7, 3
"""

from ctypes import *
from dwfconstants import *
import math
import sys

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

print("Configuring Digital In...")

nSamples = 1000000 
rgbSamples = (c_uint8*nSamples)()
cAvailable = c_int(0)
cLost = c_int(0)
cCorrupt = c_int(0)

idxCS = [0,4] # DIO-0 or 24, DIO-4 or 28
idxCL = [1,5] # DIO-1 or 25, DIO-5 or 29
idxDT = [2,6] # DIO-2 or 26, DIO-6 or 30
nBits = 8

# record mode
dwf.FDwfDigitalInAcquisitionModeSet(hdwf, acqmodeRecord)
# for sync mode set divider to -1 
dwf.FDwfDigitalInDividerSet(hdwf, -1)
# 8bit per sample format, DIO 0-7
dwf.FDwfDigitalInSampleFormatSet(hdwf, 8)
# for Digital Discovery 
# with order 0: DIN0:7;   with 32 bit sampling [DIN0:23  + DIO24:31]
# with order 1: DIO24:31; with 32 bit sampling [DIO24:39 + DIN0:15]
dwf.FDwfDigitalInInputOrderSet(hdwf, 1)
# continuous sampling 
dwf.FDwfDigitalInTriggerPositionSet(hdwf, -1)
# in sync mode the trigger is used for sampling condition
# trigger detector mask:          low &     high & ( rising | falling )
mask = (1<<idxCL[0])|(1<<idxCL[1])|(1<<idxCS[0])|(1<<idxCS[1])
dwf.FDwfDigitalInTriggerSet(hdwf, 0, 0, mask, mask)
# sample on clock rising edge for sampling bits, or CS rising edge to detect frames

print("Press Ctrl+C to stop...")
dwf.FDwfDigitalInConfigure(hdwf, 1, 1)

try:
    fsDT = [0,0]
    cBit = [0,0]
    nCL = [0,0]
    pCL = [0,0]
    cWord = [0,0]
    fDT = [open("spi1.csv", "w"), open("spi2.csv", "w")]
    while True:
        if dwf.FDwfDigitalInStatus(hdwf, 1, byref(sts)) != 1:
            szerr = create_string_buffer(512)
            dwf.FDwfGetLastErrorMsg(szerr)
            print(szerr.value)
            break
        if sts.value != DwfStateTriggered.value :
            continue
        dwf.FDwfDigitalInStatusRecord(hdwf, byref(cAvailable), byref(cLost), byref(cCorrupt))
        if cLost.value :
            print("Samples were lost!")
            break
        if cCorrupt.value :
            print("Samples could be corrupt!")
            break
        if cAvailable.value == 0 :
            continue
        if cAvailable.value > nSamples :
            print("Software buffer overflow")
            break
        dwf.FDwfDigitalInStatusData(hdwf, rgbSamples, c_int(cAvailable.value))
        for spi in range(2):
            for i in range(cAvailable.value):
                v = rgbSamples[i]
                if (v>>idxCS[spi])&1 : # CS high, inactive
                    if cBit[spi] != 0: # log leftover bits, frame not multiple of nBits
                        print("DT%d left h%X %d" % (spi+1, fsDT[spi], cBit[spi])) # remove print for better performance
                        fDT[spi].write("%X %d" % (fsDT[spi], cBit[spi]))
                    elif cWord[spi]:
                        fDT[spi].write("\n")
                    cBit[spi] = 0
                    fsDT[spi] = 0
                    cWord[spi] = 0
                    continue
                nCL[spi] = (v>>idxCL[spi])&1
                if nCL[spi]==1 and pCL[spi]==0: # clock rising edge
                    cBit[spi] += 1
                    fsDT[spi] <<= 1 # MSB first
                    if (v>>idxDT[spi])&1 :
                        fsDT[spi] |= 1
                    if cBit[spi] >= nBits: # got nBits of bits
                        print("DT%d h%02X" % (spi+1, fsDT[spi])) # remove print for better performance
                        fDT[spi].write("%02X " % fsDT[spi])
                        cBit[spi] = 0
                        fsDT[spi] = 0
                        cWord[spi] += 1
                pCL[spi] = nCL[spi]
except KeyboardInterrupt: # Ctrl+C
    pass

dwf.FDwfDeviceClose(hdwf)

fDT[0].close()
fDT[1].close()








