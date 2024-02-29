import vgamepad as vg
import time
import numpy as np

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

# since gamepad inputs are a bit different than HOTAS inputs, we need to define a curve function to make small inputs on the gamepad more sensitive
# the small values closer to zero needs to be larger, the larger values closer to 1 needs to be smaller
def custom_sigmoid(x, steepness=4, x_mid=0):
    return 2 * (1 / (1 + np.exp(-steepness * (x - x_mid)))) - 1

print(custom_sigmoid(0))
print(custom_sigmoid(0.1))
print(custom_sigmoid(0.2))
print(custom_sigmoid(0.3))
print(custom_sigmoid(0.4))
print(custom_sigmoid(0.5))
print(custom_sigmoid(0.6))
print(custom_sigmoid(0.7))
print(custom_sigmoid(0.8))
print(custom_sigmoid(0.9))
exit()

print("Waiting 3 seconds for initialization...")
time.sleep(3)

print("Testing gamepad...")
# test pitch axis:
gamepad.left_joystick_float(0, -0.1)
gamepad.update()
time.sleep(10)
gamepad.left_joystick_float(0, 0)
gamepad.update()
time.sleep(1)
