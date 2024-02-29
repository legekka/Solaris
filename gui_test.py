from modules.visualize import GUI
import time

gui = GUI()

for i in range(100):
    gui.update_and_draw({
        "primary_fire": 0,
        "secondary_fire": 0,
        "frameshift_jump": 0,
        "throttle": i/100,
        "frequency": 0,
        "pitch": 0,
        "yaw": 0,
        "roll": 0,
        "fss_pitch": 0,
        "fss_yaw": 0,
        "fss_zoom_out": 0,
        "fss_mode": 0,
        "return_from_surface_scanner": 0
    })
    
    time.sleep(0.1)