import time
from pylablib.devices import Thorlabs

class Motor:
    """
    Handles motor control.
    """

    def __init__(
        self,
        serialNoYaw,
        serialNoPitch,
        step_yaw,
        step_pitch,
    ):
        """Initialize motors.

        Keyword arguments:
        serialNoYaw, serialNoPitch -- Serial numbers corresponding to each ThorLabs kdc101 controller.
        yawBoundary, pitchBoundary -- Hardcoded values beyond which the motor's limit switch will trigger.
        deviationVal -- The signal intensity percentage which PlotField uses as its field boundaries.
        startYaw, startPitch -- The initial homed yaw and pitch, reset when recenterHome is called. Note: not the motors' home values.
        loc_log -- Text file where timestamped motor positions are output after each move.
        """
        self.deviceYaw = Thorlabs.KinesisMotor(serialNoYaw)
        self.devicePitch = Thorlabs.KinesisMotor(serialNoPitch)
        time.sleep(0.5)
        self.yaw_locs = []
        self.pitch_locs = []
        self.step_yaw = step_yaw
        self.step_pitch = step_pitch
        
    def safeMove(self, deviceName, amount, velocity=79475.19421577454):
        """A move function that also logs locations and displays in the GUI, then waits for the move to complete."""
        if deviceName == "yaw":
            device = self.deviceYaw
        elif deviceName == "pitch":
            device = self.devicePitch
        else:
            raise Exception("Invalid device name")
        device.setup_velocity(max_velocity=velocity)
        device.move_by(amount)
        device.wait_move()
        yaw_loc = self.deviceYaw.get_position()
        pitch_loc = self.devicePitch.get_position()
        self.yaw_locs.append(yaw_loc * self.step_yaw)
        self.pitch_locs.append(pitch_loc * self.step_pitch)
    
    def getPosition(self, deviceName):
        if deviceName == "yaw":
            return self.deviceYaw.get_position
        elif deviceName == "pitch":
            return self.devicePitch.get_position
        else:
            raise Exception("Invalid device name")
    
    def setupVelocity(self, deviceName, velocity):
        if deviceName == "yaw":
            self.deviceYaw.setup_velocity(max_velocity=velocity)
        elif deviceName == "pitch":
            self.devicePitch.setup_velocity(max_velocity=velocity)
        else:
            raise Exception("Invalid device name")
    
    def setupLimitSwitch(self, deviceName, sw_kind, sw_position_cw=None, sw_position_ccw=None):
        if deviceName == "yaw":
            device=self.deviceYaw
        elif deviceName == "pitch":
            device=self.devicePitch
        else:
            raise Exception("Invalid device name")
        device.setup_limit_switch(sw_kind=sw_kind, sw_position_cw=sw_position_cw, sw_position_ccw=sw_position_ccw)
        
    def getScale(self, deviceName):
        if deviceName == "yaw":
            return(self.deviceYaw.get_scale())
        elif deviceName == "pitch":
            return(self.devicePitch.get_scale())
        else:
            raise Exception("Invalid device name")
    def getScaleUnits(self, deviceName):
        if deviceName == "yaw":
            return(self.deviceYaw.get_scale_units())
        elif deviceName == "pitch":
            return(self.devicePitch.get_scale_units())
        else:
            raise Exception("Invalid device name")
        
    def shutDown(self):
        self.deviceYaw.setup_limit_switch(sw_kind="ignore")
        self.devicePitch.setup_limit_switch(sw_kind="ignore")
        self.deviceYaw.stop()
        self.devicePitch.stop()
        self.deviceYaw.close()
        self.devicePitch.close()
