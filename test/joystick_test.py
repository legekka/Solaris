import pygame

pygame.init()
joystick_count = pygame.joystick.get_count()
print("Joystick count: " + str(joystick_count))

# list all joysticks
for i in range(joystick_count):
    joystick = pygame.joystick.Joystick(i)
    joystick.init()
    print("Joystick " + str(i) + " name: " + joystick.get_name())

    # get the number of axes and buttons
    axes = joystick.get_numaxes()
    print("Joystick " + str(i) + " axes: " + str(axes))
    buttons = joystick.get_numbuttons()
    print("Joystick " + str(i) + " buttons: " + str(buttons))
    hats = joystick.get_numhats()
    print("Joystick " + str(i) + " hats: " + str(hats))

    # get the current position of the axes
    for i in range(axes):
        axis = joystick.get_axis(i)
        print("Axis " + str(i) + " value: " + str(axis))

    # get the current state of the buttons
    for i in range(buttons):
        button = joystick.get_button(i)
        print("Button " + str(i) + " value: " + str(button))

    # get the current state of the hats
    for i in range(hats):
        hat = joystick.get_hat(i)
        print("Hat " + str(i) + " value: " + str(hat))
