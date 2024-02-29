# this modul visualizes the output values of the model (the 13 classes) using pygame and GUI

import pygame

class GUI():
    def __init__(self):
        # init pygame
        if not pygame.get_init():
            pygame.init()
        self.width = 800
        self.height = 600
        self.win = pygame.display.set_mode((self.width, self.height))
        # get font list
        pygame.font.init()
        self.font = pygame.font.SysFont("Calibri", 12)
        self.clock = pygame.time.Clock()
        self.run = True

        self.primary_fire = 0
        self.secondary_fire = 0
        self.frameshift_jump = 0
        self.throttle = 0
        self.frequency = 0
        self.pitch = 0
        self.yaw = 0
        self.roll = 0
        self.fss_pitch = 0
        self.fss_yaw = 0
        self.fss_zoom_out = 0
        self.fss_mode = 0
        self.return_from_surface_scanner = 0

    def _remap_axis(self, value):
        # Remaps the value from -1, 1 to 0, 1
        output = (value + 1) / 2
        # clamp the value between 0 and 1
        return max(0, min(1, output))

    def _remap_button(self, value):
        # Based on value is smaller than 0.5, return 0, else return 1
        return 1 if value > 0.5 else 0
    
    def update_and_draw(self, input_dict):
        self.update(input_dict)
        self.draw()

    def update(self, output):
        self.primary_fire = self._remap_button(output["primary_fire"])
        self.secondary_fire = self._remap_button(output["secondary_fire"])
        self.frameshift_jump = self._remap_button(output["frameshift_jump"])
        self.throttle = self._remap_axis(-output["throttle"]) # inverted
        self.frequency = self._remap_axis(output["frequency"])
        self.pitch = self._remap_axis(-output["pitch"]) # pitch only needs to be inverted because of the coordinate system
        self.yaw = self._remap_axis(output["yaw"])
        self.roll = self._remap_axis(-output["roll"]) # roll input is also inverted
        self.fss_pitch = self._remap_axis(output["fss_pitch"])
        self.fss_yaw = self._remap_axis(output["fss_yaw"])
        self.fss_zoom_out = self._remap_button(output["fss_zoom_out"])
        self.fss_mode = self._remap_button(output["fss_mode"])
        self.return_from_surface_scanner = self._remap_button(output["return_from_surface_scanner"])

    def draw(self):
        self.win.fill((255, 255, 255))
        # draw the output
        self.draw_output()
        pygame.display.update()

    def draw_output(self):
        # Throttle is a slider
        self.draw_slider("Throttle", self.throttle, 100, 100)
        # Frequency is a slider as well
        self.draw_slider("Frequency", self.frequency, 100, 200)
        # For Pitch and Yaw, we use a circle which is translated by the values
        self.draw_circle(["Yaw", "Pitch"], [self.yaw, self.pitch], 400, 100)
        # Roll is a slider under the circle
        self.draw_slider("Roll", self.roll, 100, 300)
        # Primary fire and secondary fire are boxes
        self.draw_box("Primary Fire", self.primary_fire, 100, 400)
        self.draw_box("Secondary Fire", self.secondary_fire, 400, 400)
        # Frameshift jump is a box as well
        self.draw_box("Frameshift Jump", self.frameshift_jump, 100, 500)
        # the others will be implemented later

    def draw_slider(self, name, value, x, y):
        # draw the name
        text = self.font.render(name, 1, (0, 0, 0))
        self.win.blit(text, (x, y))
        # draw the slider
        pygame.draw.rect(self.win, (0, 0, 0), (x, y + 50, 200, 20))
        pygame.draw.rect(self.win, (255, 0, 0), (x, y + 50, int(value * 200), 20))

    def draw_circle(self, names, values, x, y):
        # draw the names
        for i, name in enumerate(names):
            text = self.font.render(name, 1, (0, 0, 0))
            self.win.blit(text, (x + i * 200, y))
        # draw the circle
        pygame.draw.circle(self.win, (0, 0, 0), (x + 100, y + 100), 100)
        # draw the smaller translated inner circle
        # don't forget that values are between 0 and 1, and 0.5 is the center
        dotX = int(x + 100 + (values[0] - 0.5) * 100)
        dotY = int(y + 100 + (values[1] - 0.5) * 100)
        pygame.draw.circle(self.win, (255, 0, 0), (dotX, dotY), 10) 

    def draw_box(self, name, value, x, y):
        # draw the name
        text = self.font.render(name, 1, (0, 0, 0))
        self.win.blit(text, (x, y))
        # draw the box
        pygame.draw.rect(self.win, (0, 0, 0), (x, y + 25, 200, 50))
        if value == 1:
            pygame.draw.rect(self.win, (255, 0, 0), (x, y + 25, 200, 50))

    def close(self):
        self.run = False
        pygame.quit()

