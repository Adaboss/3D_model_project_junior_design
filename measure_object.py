import cv2
import numpy as np

# --- Configuration ---
CHECKERBOARD = (11, 14)  
SQUARE_SIZE_MM = 8.0   # Size of a single square in mm
MIN_OBJECT_AREA = 1000 # Minimum area for objects to be measured (in pixels)
EMA_ALPHA = 0.05       # Smoothing factor for jitter reduction (lower = smoother)

def calibrate_and_measure():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    objpoints = [] 
    imgpoints = [] 
    
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_MM

    print("=== CAMERA CALIBRATION & MEASUREMENT ===")
    print("Press 'c' to capture a frame for Lens Distortion calibration.")
    print("Press 'k' to calculate camera matrices and switch to measuring.")
    print("Press 's' to SKIP calibration entirely (Not recommended if you need high accuracy).")
    print("Press 'q' at any time to quit.")
    print("========================================")
    
    calibrated = False
    camera_matrix = None
    dist_coeffs = None
    
    pixels_per_mm = None
    smooth_pixels_per_mm = None # Used to stop the scale from jittering
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display_frame = frame.copy()
            
            if not calibrated:
                # Mode 1: Lens Calibration phase
                ret_cb, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
                
                if ret_cb:
                    cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, ret_cb)
                    
                cv2.putText(display_frame, f"Captured frames: {len(objpoints)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, "Press 'c' to capture (Board must be fully visible)", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, "Press 'k' to finish calibration", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, "Press 's' to SKIP (Warning: Uncalibrated mapping)", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                            
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    if ret_cb:
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                        imgpoints.append(corners_refined)
                        objpoints.append(objp)
                        print(f"Captured! Total frames gathered: {len(objpoints)}")
                    else:
                        print(f"WARNING: Checkerboard with exactly {CHECKERBOARD} inner corners NOT FULLY detected.")
                        
                elif key == ord('k'):
                    if len(objpoints) > 0:
                        print("Calibrating camera... Please wait.")
                        ret_calib, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                            objpoints, imgpoints, gray.shape[::-1], None, None)
                        print("Calibration successful!")
                        calibrated = True
                    else:
                        print("Capture at least 1 frame (press 'c') before calibrating, or press 's' to skip.")
                        
                elif key == ord('s'):
                    print("Skipping Lens Calibration. Proceeding directly to measurement mode...")
                    calibrated = True
                    camera_matrix = None
                    
                elif key == ord('q'):
                    break
                    
            else:
                # Mode 2: Measurement phase
                if camera_matrix is not None:
                    h, w = frame.shape[:2]
                    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
                    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
                    x, y, w_roi, h_roi = roi
                    undistorted = undistorted[y:y+h_roi, x:x+w_roi]
                else:
                    undistorted = frame.copy()
                
                gray_undistorted = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
                
                # --- Dynamic Square Detection for Scale ---
                blurred_scale = cv2.GaussianBlur(gray_undistorted, (5, 5), 0)
                thresh_scale = cv2.adaptiveThreshold(blurred_scale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                cnts_scale, _ = cv2.findContours(thresh_scale, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                
                square_sizes = []
                for c in cnts_scale:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                    
                    if len(approx) == 4:
                        area = cv2.contourArea(c)
                        if 100 < area < 15000:  
                            # Use minAreaRect for high-precision floating point dimensions rather than crude integer bounding box
                            rect_sq = cv2.minAreaRect(approx)
                            (cx, cy), (w_sq, h_sq), angle = rect_sq
                            
                            if w_sq == 0 or h_sq == 0:
                                continue
                                
                            aspect_ratio = float(w_sq) / float(h_sq)
                            
                            if 0.85 <= aspect_ratio <= 1.15:
                                # We found a plausible square!
                                square_sizes.append((w_sq + h_sq) / 2.0) # Average of width and height for accuracy
                                cv2.drawContours(undistorted, [approx], -1, (0, 0, 255), 1) 
                                
                if len(square_sizes) >= 3:
                    median_pixel_size = np.median(square_sizes)
                    current_ppm = median_pixel_size / SQUARE_SIZE_MM
                    
                    # Apply Exponential Moving Average (EMA) to completely kill the jitter
                    if smooth_pixels_per_mm is None:
                        smooth_pixels_per_mm = current_ppm
                    else:
                        smooth_pixels_per_mm = (EMA_ALPHA * current_ppm) + ((1.0 - EMA_ALPHA) * smooth_pixels_per_mm)
                    
                    pixels_per_mm = smooth_pixels_per_mm
                    
                    cv2.putText(undistorted, f"Scale: {smooth_pixels_per_mm:.2f} px/mm", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    if smooth_pixels_per_mm is None:
                        cv2.putText(undistorted, "No 8mm squares detected for scale", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(undistorted, f"Scale (cached): {smooth_pixels_per_mm:.2f} px/mm", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # --- Object Detection & Measurement ---
                if smooth_pixels_per_mm is not None:
                    blurred = cv2.GaussianBlur(gray_undistorted, (7, 7), 0)
                    edged = cv2.Canny(blurred, 50, 100)
                    edged = cv2.dilate(edged, None, iterations=1)
                    edged = cv2.erode(edged, None, iterations=1)
                    
                    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for c in cnts:
                        if cv2.contourArea(c) < MIN_OBJECT_AREA:
                            continue
                            
                        rect = cv2.minAreaRect(c)
                        (ctrx, ctry), (width_px, height_px), angle = rect
                        
                        width_mm = width_px / smooth_pixels_per_mm
                        height_mm = height_px / smooth_pixels_per_mm
                        
                        if width_mm > height_mm:
                            width_mm, height_mm = height_mm, width_mm
                            
                        # Ignore the individual 8mm checkerboard squares themselves
                        if (6.5 < width_mm < 9.5) and (6.5 < height_mm < 9.5):
                            continue
                        
                        box = cv2.boxPoints(rect)
                        box = np.int32(box)
                        cv2.drawContours(undistorted, [box], 0, (0, 255, 0), 2)
                        
                        cv2.putText(undistorted, f"{width_mm:.1f}x{height_mm:.1f}mm", 
                                    (int(ctrx - 40), int(ctry + 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                display_frame = undistorted
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    calibrated = False
                    objpoints = []
                    imgpoints = []
                    pixels_per_mm = None
                    smooth_pixels_per_mm = None
                    camera_matrix = None
                    print("Returning to calibration mode...")

            cv2.imshow("Calibration & Measurement", display_frame)
            
    except KeyboardInterrupt:
        print("\n[INFO] Program forcefully closed via Ctrl+C. Releasing camera buffer immediately...")
    finally:
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera buffer gracefully closed.")

if __name__ == "__main__":
    calibrate_and_measure()
