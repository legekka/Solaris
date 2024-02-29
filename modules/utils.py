from win32gui import FindWindow, GetWindowRect
import mss
from PIL import Image

def _get_elite_window_rect():
    window_handle = FindWindow(None, "Elite - Dangerous (CLIENT)")
    window_rect = GetWindowRect(window_handle)
    window_rect = (window_rect[0]+8, window_rect[1]+31, window_rect[2], window_rect[3])
    window_rect = (window_rect[0], window_rect[1], window_rect[0]+1280, window_rect[1]+720)
    return window_rect

def get_elite_screenshot():
    window_rect = _get_elite_window_rect()
    with mss.mss() as sct:
        screenshot = sct.grab(window_rect)

        # convert to PIL image
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        return img
    