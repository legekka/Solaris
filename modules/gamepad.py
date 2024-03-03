import vgamepad as vg
import numpy as np

class Gamepad:
    def __init__(self):
        self.gamepad = vg.VX360Gamepad()
        
        # initialize axes. Axes are between -1 and 1 
        self.throttle = 0               # Right Trigger
        self.frequency = 0              # Right Joystick X
        self.pitch = 0                  # Left Joystick Y
        self.yaw = 0                    # Right Joystick X
        self.roll = 0                   # Left Joystick X
        self.fss_yaw = 0                # Left Joystick X
        self.fss_pitch = 0              # Left Joystick Y

        # initialize buttons
        self.primary_fire = 0           # Right Bumper
        self.secondary_fire = 0         # Left Bumper
        self.frameshift_jump = 0        # Y buttong
        self.return_from_surface_scanner = 0
        self.fss_zoom_out = 0
        self.fss_mode = 0

    def _remap_axis(self, value):
        # Remaps the value from -1, 1 to 0, 1
        output = (value + 1) / 2
        # clamp the value between 0 and 1
        return max(0, min(1, output))
    
    def _clamp_axis(self, value):
        # Clamp the value between -1 and 1
        return max(-1, min(1, value))

    def _remap_button(self, value):
        # Based on value is smaller than 0.5, return 0, else return 1
        return 1 if value > 0.5 else 0
    
    def _custom_sigmoid(self, x, steepness=3, x_mid=0):
        return 2 * (1 / (1 + np.exp(-steepness * (x - x_mid)))) - 1

    def update(self, input_dict):
        """
        Update the gamepad with the input values
        
        Args:
        input_dict: a dictionary containing the changed input values
        """

        for key, value in input_dict.items():
            if key == "throttle":
                self.throttle = value
                self.gamepad.right_trigger_float(self._remap_axis(-value)) # inverted
            elif key == "frequency":
                self.frequency = value
                #self.gamepad.right_joystick_float(self._remap_axis(value), 0)
            elif key == "pitch":
                self.pitch = value
            elif key == "yaw":
                self.yaw = value
                yaw = self._custom_sigmoid(self._clamp_axis(self.yaw))
                self.gamepad.right_joystick_float(yaw, 0)
            elif key == "roll":
                self.roll = value
            elif key == "fss_yaw":
                self.fss_yaw = value
                #self.gamepad.left_joystick_float(self._remap_axis(value), 0)
            elif key == "fss_pitch":
                self.fss_pitch = value
                #self.gamepad.left_joystick_float(0, self._remap_axis(value))
            elif key == "primary_fire":
                self.primary_fire = value
                value = self._remap_button(value)
                self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER) if value == 1 else self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
            elif key == "secondary_fire":
                self.secondary_fire = value
                value = self._remap_button(value)
                self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER) if value == 1 else self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
            elif key == "frameshift_jump":
                self.frameshift_jump = value
                value = self._remap_button(value)
                self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_Y) if value == 1 else self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
            elif key == "return_from_surface_scanner":
                self.return_from_surface_scanner = value
                # not implemented yet
            elif key == "fss_zoom_out":
                self.fss_zoom_out = value
                # not implemented yet
            elif key == "fss_mode":
                self.fss_mode = value
                # not implemented yet

        pitch = self._custom_sigmoid(self._clamp_axis(-self.pitch))
        roll = self._custom_sigmoid(self._clamp_axis(-self.roll))
        self.gamepad.left_joystick_float(roll, pitch) # inverted
        # update the gamepad
        self.gamepad.update()