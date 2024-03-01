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
# test throttle axis:
# gamepad.right_trigger_float(1)
# gamepad.update()
# time.sleep(1)
# gamepad.right_trigger_float(0)
# gamepad.update()
# time.sleep(1)

# test pitch axis:
# gamepad.left_joystick_float(0, 1)
# gamepad.update()
# time.sleep(1)
# gamepad.left_joystick_float(0, 0)
# gamepad.update()
# time.sleep(1)

# test roll axis:
# gamepad.left_joystick_float(1, 0)
# gamepad.update()
# time.sleep(1)
# gamepad.left_joystick_float(0, 0)
# gamepad.update()
# time.sleep(1)

# test yaw axis:
# gamepad.right_joystick_float(1, 0)
# gamepad.update()
# time.sleep(1)
# gamepad.right_joystick_float(0, 0)
# gamepad.update()
# time.sleep(1)


# test primary fire button (right bumper)
# gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
# gamepad.update()
# time.sleep(0.2)
# gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
# gamepad.update()
# time.sleep(1)

# test secondary fire button (left bumper)
# gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
# gamepad.update()
# time.sleep(0.2)
# gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
# gamepad.update()
# time.sleep(1)


# test frameshift jump button (Y button)
gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
gamepad.update()
time.sleep(0.2)
gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
gamepad.update()
time.sleep(1)
exit()

# gamepad.right_joystick_float(0, 1)
# gamepad.update()
# time.sleep(1)
# gamepad.right_joystick_float(0, 0)
# gamepad.update()
# time.sleep(1)
