import cv2
import time
from gpiozero import OutputDevice

# Define GPIO pins connected to the A4988
DIR_PIN = 20
STEP_PIN = 21

direction = OutputDevice(DIR_PIN)
step = OutputDevice(STEP_PIN)

def move_stepper(steps, delay, clockwise=True):
    if clockwise:
        direction.on()
    else:
        direction.off()
        
    for _ in range(steps):
        step.on()
        time.sleep(delay)
        step.off()
        time.sleep(delay)

def main():
    print("Initializing webcam for alignment...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return
        
    print("=======================================")
    print("Webcam activated. A live video window should appear.")
    print("Select the video window, then use these controls:")
    print("  [ d ] - Rotate turntable Clockwise slightly")
    print("  [ a ] - Rotate turntable Counter-Clockwise slightly")
    print("  [ c ] - Capture reference photo (saves to alignment.jpg)")
    print("  [ q ] - Quit alignment tool")
    print("=======================================")
    
    cv2.namedWindow("Alignment Tool (Press Q to quit)", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam. Retrying...")
                time.sleep(0.1)
                continue
            
            # Draw a center crosshair to help with keeping the object perfectly centered!
            height, width = frame.shape[:2]
            cx, cy = width // 2, height // 2
            
            # Add semi-transparent crosshair (Vertical & Horizontal lines)
            cv2.line(frame, (cx, 0), (cx, height), (0, 255, 0), 1)
            cv2.line(frame, (0, cy), (width, cy), (0, 255, 0), 1)
            
            # Draw a circle in the very middle
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), 1)
            
            cv2.imshow("Alignment Tool (Press Q to quit)", frame)
            
            # Wait for keypress
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27: # q or ESC
                print("Exiting tool.")
                break
            elif key == ord('d'):
                move_stepper(10, 0.01, clockwise=True) # 10 steps = 18 degrees usually
            elif key == ord('a'):
                move_stepper(10, 0.01, clockwise=False)
            elif key == ord('c'):
                cv2.imwrite("alignment_test.jpg", frame)
                print("Saved reference picture to alignment_test.jpg!")
                
    except Exception as e:
        print(f"An error occurred: {e}")
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        direction.off()
        step.off()

if __name__ == "__main__":
    main()
