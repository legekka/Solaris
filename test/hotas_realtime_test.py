import pygame

def InitJoy(name):
        joystick_count = pygame.joystick.get_count()
        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            if name in joystick.get_name():
                break
        return joystick
    

pygame.init()

global hotas_throttle, hotas_stick

# find hotas devices by name
hotas_throttle = InitJoy("Throttle")
hotas_stick = InitJoy("Stick")
# print hotas devices
print("Hotas throttle: " + str(hotas_throttle.get_name()))
print("Hotas stick: " + str(hotas_stick.get_name()))


import time 

# print all the values (axis, buttons, etc) of the hotas throttle
while True:
    pygame.event.pump()
    # print("Throttle")
    # for i in range(hotas_throttle.get_numaxes()):
    #     axis = hotas_throttle.get_axis(i)
    #     print("Axis " + str(i) + " value: " + str(axis))
    # print("---\nStick\n")
    # for i in range(hotas_stick.get_numaxes()):
    #     axis = hotas_stick.get_axis(i)
    #     print("Axis " + str(i) + " value: " + str(axis))
    
    # print stick buttons
    print("Stick buttons")
    for i in range(hotas_stick.get_numbuttons()):
        button = hotas_stick.get_button(i)
        print("Button " + str(i) + " value: " + str(button))

    time.sleep(0.25)
    print("\n")