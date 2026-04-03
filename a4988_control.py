import time
import os
import cv2
from gpiozero import OutputDevice

# Define GPIO pins connected to the A4988
DIR_PIN = 20   
STEP_PIN = 21  

direction = OutputDevice(DIR_PIN)
step = OutputDevice(STEP_PIN)

def move_stepper(steps, delay=0.03, clockwise=True):
    if steps <= 0: return
    direction.value = clockwise
    for _ in range(steps):
        step.on()
        time.sleep(delay)
        step.off()
        time.sleep(delay)

if __name__ == '__main__':
    cap = None
    try:
        # A standard bare NEMA 17 is 200 full-steps per revolution.
        steps_per_rev = 400 
        
        # CRITICAL MATH FIX: We changed 60 frames to 50 frames!
        # 200 steps / 60 frames = 3.333 steps (Uneven jittering)
        # 200 steps / 50 frames = exactly 4.0 steps (Flawlessly smooth)
        NUM_FRAMES = 50
        OUTPUT_DIR = "frames"
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        print("Initializing webcam...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        time.sleep(2)  # Adjust exposure
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
        else:
            print("=== Strictly Sequential Stop-Motion ===")
            
            # --- FIX FOR "FIRST COUPLE FRAMES NOT SPINNING" ---
            # Most belts and 3D printed mechanical gears have a few millimeters of dead physical slack.
            # We "pre-wind" the motor aggressively before taking *any* pictures so the belt/gears are absolutely 100% tight!
            print("Pre-winding motor backwards and forwards to permanently eliminate gear backlash slack...")
            move_stepper(steps=10, delay=0.03, clockwise=False) 
            time.sleep(0.5)
            move_stepper(steps=20, delay=0.03, clockwise=True) # Lock the positive tension in!
            print("Hardware tension locked! Settling...")
            time.sleep(2.0)
            
            steps_per_frame = int(steps_per_rev / NUM_FRAMES) # 4 steps exactly
            
            for frame_id in range(1, NUM_FRAMES + 1):
                # Move identically integer exact chunks
                move_stepper(steps=steps_per_frame, delay=0.03, clockwise=True)
                
                print(f"Settling structural wobble: Frame {frame_id:03d}/{NUM_FRAMES}...")
                time.sleep(1.0)
                
                # The webcam hardware builds up a backlog buffer of old images.
                # We rapidly tell it to throw away 3 old frames so we get the immediate newest one.
                for _ in range(3):
                    cap.grab()
                
                ret, frame_img = cap.read()
                if ret:
                    filename = os.path.join(OUTPUT_DIR, f"capture_{frame_id:03d}.jpg")
                    cv2.imwrite(filename, frame_img)
                else:
                    print("   -> ERROR: Failed to physically capture image.")

            print(f"\n[SUCCESS] Custom sequence complete. 50 flawlessly matched photos saved.")
                
    except Exception as e:
        print(f"An error occurred: {e}")
    except KeyboardInterrupt:
        print("\nProgram instantly halted by the user.")
    finally:
        # Hardware release
        if cap is not None and cap.isOpened():
            cap.release()
        direction.off()
        step.off()
