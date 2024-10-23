"""
Import files from local system
"""
dwf = ctypes.cdll.dwf

HERE = Path(__file__).parent
path.append(str(HERE / "py/"))
path.append("py")
# import dwfconstants as constants
from WF_SDK import device, scope, wavegen, error  # import instruments

modulename = "WF_SDK"
if modulename not in sys.modules:
    print(f"You have not imported the {modulename} module")
    

class data:
    """stores the sampling frequency and the buffer size for Digilent to use"""

    sampling_frequency = 8.192e04
    buffer_size = 8192
    total_time = 0.1

class Scope:
    """
    Handles spectrum observation.
    """

    def __enter__(self):
        """Allows for scope to be properly spun down and closed using __exit__ even if program is interrupted."""
        return self

    def __init__(
        self,
    ):
        """Initialize scope.        """
        """ Run digilent """
        self.device_data = device.open()
        print("digilent device opened")
        scope.open(
            self.device_data,
            sampling_frequency=8.192e04,
            buffer_size=8192,
            amplitude_range=50,
        )
        scope.trigger(
            self.device_data,
            enable=False,
            source=scope.trigger_source.analog,
            channel=1,
            level=0,
        )
        print("scope set up")
    
    def recordData(self):
        return scope.record(self.device_data, channel=1)

    def shutDown(self):
        scope.close(self.device_data)
        device.close(self.device_data)
