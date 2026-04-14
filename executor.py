import ctypes
import time

# Windows Virtual Key Codes
VK_VOLUME_UP = 0xAF
VK_VOLUME_DOWN = 0xAE
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3

class OSExecutor:
    @staticmethod
    def press_key(hex_code):
        ctypes.windll.user32.keybd_event(hex_code, 0, 0, 0) # Key Down
        time.sleep(0.05)
        ctypes.windll.user32.keybd_event(hex_code, 0, 2, 0) # Key Up

ACTIONS = {
    "Volume Up": VK_VOLUME_UP,
    "Volume Down": VK_VOLUME_DOWN,
    "Next Track": VK_MEDIA_NEXT_TRACK,
    "Prev Track": VK_MEDIA_PREV_TRACK,
    "Play/Pause": VK_MEDIA_PLAY_PAUSE
}