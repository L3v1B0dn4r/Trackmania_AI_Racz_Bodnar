import mss
import numpy as np
import cv2

monitor = {"top":200, "left": 100, "width": 800, "height": 600}

with mss.mss() as sct:
    while True:
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        cv2.imshow("AI View", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cv2.destroyAllWindows()