# this module defines the classes for the Logitech X56 HOTAS Throttle and Stick

import pygame

class Throttle():
    def __init__(self):
        # init pygame if it's not already
        if not pygame.get_init():
            pygame.init()

        joystick_count = pygame.joystick.get_count()
        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            if "Throttle" in joystick.get_name():
                break
        self.joystick = joystick

        # we have 7 axes on the throttle controller
        # 0,1: is the throttle, it's two axes, but we treat it as one - it also has similar values
        # 2: Frequency dial
        # we don't use the others

        # for buttons:
        # 11: Frameshift Jump
        # 2: return from Surface Scanner
        
    def update(self):
        pygame.event.pump()
        # get the current position of the axes, round it to 2 decimal places
        self.throttle = round(self.joystick.get_axis(0), 2)
        self.frequency = round(self.joystick.get_axis(2), 2)
        # get the current state of the buttons
        self.frameshift_jump = self.joystick.get_button(11)
        self.return_from_surface_scanner = self.joystick.get_button(2)

class Stick():
    def __init__(self):
        pygame.init()
        joystick_count = pygame.joystick.get_count()
        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            if "Stick" in joystick.get_name():
                break
        self.joystick = joystick

        # we have 5 axes on the stick controller
        # 0: Roll (roll left/roll right)
        # 1: Pitch (up/down)
        # 2: FSS Yaw (left/right)
        # 3: FSS Pitch (up/down)
        # 4: Yaw (left/right)

        # for buttons:
        # 0: Primary Fire, Discovery Scanner (HONK), Zoom in FSS
        # 1: Secondary Fire, Surface Scanner
        # 2: Target Select (not in use)
        # 3: FSS Zoom out
        # 4: FSS Mode (on/off)

    def update(self):
        pygame.event.pump()
        # get the current position of the axes, round it to 2 decimal places
        self.roll = round(self.joystick.get_axis(0), 2)
        self.pitch = round(self.joystick.get_axis(1), 2)
        self.fss_yaw = round(self.joystick.get_axis(2), 2)
        self.fss_pitch = round(self.joystick.get_axis(3), 2)
        self.yaw = round(self.joystick.get_axis(4), 2)
        # get the current state of the buttons
        self.primary_fire = self.joystick.get_button(0)
        self.secondary_fire = self.joystick.get_button(1)
        self.target_select = self.joystick.get_button(2)
        self.fss_zoom_out = self.joystick.get_button(3)
        self.fss_mode = self.joystick.get_button(4)
