import time
import controls

print("Focus Trackmania window NOW")
time.sleep(5)

controls.accelerate()
print("Accelerating...")
time.sleep(3)

controls.left()
print("Turning left...")
time.sleep(2)

controls.release_all()
print("Done")