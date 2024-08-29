"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2022-01-19

   Requires:                       
       Python 2.7, 3
"""

from ctypes import *
from dwfconstants import *
import time
import sys

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

sTest = 5
cSamples = 64

version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: "+str(version.value))

cDevice = c_int()
dwf.FDwfEnum(0, byref(cDevice))
print("Detected Devices: "+str(cDevice.value))
for iDev in range(0, cDevice.value):
    iDevId = c_int()
    iDevRev = c_int()
    szName = create_string_buffer(64)
    szSN = create_string_buffer(16)
    dwf.FDwfEnumDeviceType(iDev, byref(iDevId), byref(iDevRev))
    dwf.FDwfEnumDeviceName(iDev, szName)
    dwf.FDwfEnumSN (iDev, szSN)
    print(str(iDev)+":\tName: " + repr(szName.value) + " " + repr(szSN.value)+"\tID: " + str(iDevId.value) + " rev: " + str(iDevRev.value))

iDev = int(input("Select device (0.."+str(cDevice.value-1)+"): "))
hdwf = c_int()
dwf.FDwfDeviceOpen(iDev, byref(hdwf))
#dwf.FDwfDeviceConfigOpen(iDev, 1, byref(hdwf))

if hdwf.value == hdwfNone.value:
    szerr = create_string_buffer(512)
    dwf.FDwfGetLastErrorMsg(szerr)
    print(szerr.value)
    print("failed to open device")
    quit()

dwf.FDwfDeviceAutoConfigureSet(hdwf, 0) # 0 = the device will only be configured when FDwf###Configure is called

#set up acquisition
dwf.FDwfAnalogInFrequencySet(hdwf, c_double(1e12))
dwf.FDwfAnalogInBufferSizeSet(hdwf, c_int(cSamples)) 
dwf.FDwfAnalogInChannelEnableSet(hdwf, 0, 1)
dwf.FDwfAnalogInConfigure(hdwf, 1, 0)

print("Generating sine wave...")
dwf.FDwfAnalogOutConfigure(hdwf, 0, c_int(1))


# after setting, wait for some time until the offset stabilizes
time.sleep(0.1)

print("Starting oscilloscope")
dwf.FDwfAnalogInConfigure(hdwf, 1, 1)

print("Testing for "+str(sTest)+" sec with "+str(cSamples)+" samples")
sts = c_byte()
rgdSamples = (c_double*cSamples)()

count = 0
start = time.time()
while time.time()-start < sTest:
    while True:
        dwf.FDwfAnalogInStatus(hdwf, 1, byref(sts))
        if sts.value == DwfStateDone.value :
            break
    dwf.FDwfAnalogInStatusData(hdwf, 0, rgdSamples, cSamples) # get channel 1 data
    count+=1
end = time.time()
dwf.FDwfDeviceCloseAll()

print("Elapsed: "+str(end-start)+" Captures: "+str(count))
print("Waveforms/second: "+str(count/(end-start))+" "+str(round(1000.0*(end-start)/count,3))+"ms")

# AD2: 3825-4408
# AD3: 3878
# ADP3450 Standard USB: 1900, EthSwitch: 2100
# ADP3450 Linux USB: 843, EthSwitch: 618
# ADP3450 Embedded AXI: 4921, EthLocal: 442

