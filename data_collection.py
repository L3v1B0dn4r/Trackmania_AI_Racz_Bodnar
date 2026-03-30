import mss
import numpy as np
import cv2
import keyboard
import time
import os

# =========================
# CONFIGURATION
# =========================
monitor = {"top": 200, "left": 100, "width": 800, "height": 600}

SAVE_FILE = "training_data.npy"
WIDTH = 160
HEIGHT = 120

# =========================
# STATE & DATA LOADING
# =========================
training_data = []
paused = True  # Start paused

if os.path.exists(SAVE_FILE):
    print("Loading existing dataset... this might take a moment.")
    training_data = list(np.load(SAVE_FILE, allow_pickle=True))
    print(f"Loaded {len(training_data)} previous samples.")

start_length = len(training_data)


# =========================
# INPUT CAPTURE (Arrow Keys)
# =========================
def get_action():
    return [
        int(keyboard.is_pressed('left')),
        int(keyboard.is_pressed('right')),
        int(keyboard.is_pressed('up')),
        int(keyboard.is_pressed('down'))
    ]


# =========================
# MAIN LOOP
# =========================
with mss.mss() as sct:
    print("===================================")
    print("DATA COLLECTION READY (PAUSED)")
    print("Press:")
    print("  P → Start/Pause Recording")
    print("  Q → Save & Quit")
    print("===================================")

    time.sleep(1)

    while True:
        # Pause toggle
        if keyboard.is_pressed('p'):
            paused = not paused
            time.sleep(0.5)  # debounce

        action = get_action()

        if not paused:
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Preprocessing
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.expand_dims(frame, axis=2)

            training_data.append([frame, action])

            # Show game preview
            cv2.imshow("Recording", frame)

        # =========================
        # STATUS WINDOW UI
        # =========================
        status_img = np.zeros((150, 300, 3), dtype=np.uint8)

        state_text = "PAUSED" if paused else "RECORDING"
        state_color = (0, 0, 255) if paused else (0, 255, 0)

        cv2.putText(status_img, state_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        cv2.putText(status_img, f"Total Frames: {len(training_data)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)

        session_frames = len(training_data) - start_length
        cv2.putText(status_img, f"Session Frames: {session_frames}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (150, 150, 150), 1)

        # Show live key presses mapped to arrows
        keys_str = f"Keys: U:{action[2]} L:{action[0]} D:{action[3]} R:{action[1]}"
        cv2.putText(status_img, keys_str, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow("Status", status_img)

        # Quit and Save
        if keyboard.is_pressed('q') or (cv2.waitKey(1) & 0xFF == ord('q')):
            print(f"Saving {len(training_data)} total frames to {SAVE_FILE}...")
            np.save(SAVE_FILE, np.array(training_data, dtype=object))
            print("Save complete! Exiting.")
            break

cv2.destroyAllWindows()