from pynput.mouse import Controller, Button
import time
mouse = Controller()

while True:
    mouse.click(Button.left, 1)
    print('click')
    time.sleep(20)