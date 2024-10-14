"""
This script is used to identify the ids of the joystick buttons, hats, and axes.
"""

import pygame
import os


class JoystickController():
    def __init__(self):
        os.environ["SDL_VIDEODRIVER"] = "dummy"  # in case you don't use a monitor
        pygame.display.init()
        pygame.joystick.init()
        detected_joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
        if len(detected_joysticks) == 0:
            print('No joystick is detected!')
            exit()
        else:
            print('Joystick detected!')
            print('Feature name: {}'.format(detected_joysticks[0].get_name()))
            self.joystick = detected_joysticks[0]
            self.joystick.init()

    def run(self):
        while True:
            pygame.event.pump()
            
            # detect button press
            for i in range(self.joystick.get_numbuttons()):
                if self.joystick.get_button(i):
                    print(f'Button {i} is pressed')

            # detect hat movement
            hat = self.joystick.get_hat(0)
            if hat != (0, 0):
                print(f'hat value: {hat}')

            # detect axis movement
            for i in range(self.joystick.get_numaxes()):
                axis_value = self.joystick.get_axis(i)
                if abs(axis_value) > 0.1 and abs(axis_value) < 0.9:
                    print(f'Axis {i} value: {axis_value}')


if __name__ == '__main__':
    joystick_controller = JoystickController()
    try:
        joystick_controller.run()
    finally:
        pygame.quit()

