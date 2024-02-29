import vgamepad as vg
import time


gamepad = vg.VX360Gamepad()

# initialize axes. Axes are between -1 and 1
throttle = 0                # Right Trigger
frequency = 0               # Right Joystick X
pitch = 0                   # Left Joystick Y
roll = 0                    # Left Joystick X
yaw = 0                     # Rigth Joystick X
fss_pitch = 0               # Left Joystick Y
fss_yaw = 0                 # Left Joystick X

# initialize buttons
primary_fire = 0            # Right Bumper
secondary_fire = 0          # Left Bumper
frameshift_jump = 0         # Y button
return_from_surface_scanner = 0
fss_zoom_out = 0
fss_mode = 0

print("Waiting 3 seconds for initialization...")
time.sleep(3)

print("Testing gamepad...")
# test roll axis:
gamepad.left_joystick_float(1, 0)
gamepad.update()
time.sleep(1)
gamepad.left_joystick_float(-1, 0)
gamepad.update()
time.sleep(1)
gamepad.left_joystick_float(0, 0)
gamepad.update()
