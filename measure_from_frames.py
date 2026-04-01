import cv2
import numpy as np
import glob
import os
import math

# --- Configuration ---
SQUARE_SIZE_MM = 8.0   # Size of a single square in mm
MIN_OBJECT_AREA = 5000 # Ignore tiny checkerboard glitches, focus on Large Arduino-sized objects
INPUT_DIR = "frames"
OUTPUT_DIR = "measured_frames"

def get_scale_from_image(gray_img, display_img):
    """
    Finds the 8mm checkerboard squares in the image and returns the scale (pixels_per_mm).
    Upgraded: Uses Otsu's Thresholding and shape Extent. It completely ignores blurriness!
    """
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    # Adaptive threshold is terrible for blurry images. Otsu's global threshold is much more stable!
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    square_sizes = []
    
    for c in cnts:
        area = cv2.contourArea(c)
        if 200 < area < 20000:  
            rect_sq = cv2.minAreaRect(c)
            (cx, cy), (w_sq, h_sq), angle = rect_sq
            
            if w_sq == 0 or h_sq == 0:
                continue
                
            aspect_ratio = float(w_sq) / float(h_sq)
            
            if 0.75 <= aspect_ratio <= 1.25:
                # Extent = area of contour / area of its bounding box. 
                # A perfect square is 1.0. A blurry square is 0.85+. A circle is 0.78.
                extent = area / (w_sq * h_sq)
                
                if extent > 0.75:
                    square_sizes.append((w_sq + h_sq) / 2.0)
                    
                    # Draw a faint red line around the valid squares found
                    box = cv2.boxPoints(rect_sq)
                    box = np.int32(box)
                    cv2.drawContours(display_img, [box], 0, (0, 0, 255), 1) 
                    
    if len(square_sizes) >= 3:
        median_pixel_size = np.median(square_sizes)
        return median_pixel_size / SQUARE_SIZE_MM
    return None

def process_offline_frames():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg"))) + sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
    
    if len(image_paths) == 0:
        print(f"Error: No images found in '{INPUT_DIR}/'")
        return
        
    print(f"Found {len(image_paths)} images. Processing...")
    
    last_known_ppm = None

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        filename = os.path.basename(img_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Get Scale
        pixels_per_mm = get_scale_from_image(gray, frame)
        if pixels_per_mm is not None:
            last_known_ppm = pixels_per_mm
        else:
            pixels_per_mm = last_known_ppm # Fallback

        # 2. Setup center coordinate tracking
        img_h, img_w = frame.shape[:2]
        center_x, center_y = img_w / 2.0, img_h / 2.0
        
        if pixels_per_mm is not None:
            cv2.putText(frame, f"Scale: {pixels_per_mm:.2f} px/mm", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
            # Object Detection
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            edged = cv2.Canny(blurred, 30, 100) # Lowered thresholds to catch faint edges in blurry photos
            
            # Dilation merges the Arduino's inner components (chips, pins, text) into ONE solid shape block!
            edged = cv2.dilate(edged, None, iterations=4)
            edged = cv2.erode(edged, None, iterations=4)
            
            cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            closest_dist = float('inf')
            best_object_rect = None
            
            for c in cnts:
                area = cv2.contourArea(c)
                # Ignore random noise. Ignore the massive container boundaries (like grabbing the whole image edge)
                if area < MIN_OBJECT_AREA or area > (img_w * img_h * 0.9):
                    continue
                    
                rect = cv2.minAreaRect(c)
                (ctrx, ctry), (width_px, height_px), angle = rect
                
                width_mm = width_px / pixels_per_mm
                height_mm = height_px / pixels_per_mm
                
                # Exclude the 8mm checkerboard squares themselves
                if (6.0 < width_mm < 10.0) and (6.0 < height_mm < 10.0):
                    continue
                
                dist_to_center = math.hypot(ctrx - center_x, ctry - center_y)
                
                # Target the shape closest to the absolute center of the photo
                if dist_to_center < closest_dist:
                    closest_dist = dist_to_center
                    best_object_rect = rect

            # 3. Draw and measure ONLY the best central object
            if best_object_rect is not None:
                (ctrx, ctry), (width_px, height_px), angle = best_object_rect
                
                width_mm = width_px / pixels_per_mm
                height_mm = height_px / pixels_per_mm
                
                if width_mm > height_mm:
                    width_mm, height_mm = height_mm, width_mm
                
                # Highlight the measured object
                box = cv2.boxPoints(best_object_rect)
                box = np.int32(box)
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 3)
                
                # Draw a crosshair indicating we locked onto the middle
                cv2.drawMarker(frame, (int(ctrx), int(ctry)), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                
                # Print dimensions
                text = f"{width_mm:.1f}x{height_mm:.1f}mm"
                cv2.putText(frame, text, (int(ctrx) - 50, int(ctry) - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)
            else:
                cv2.putText(frame, "No central object found", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
        else:
            cv2.putText(frame, "Error: Could not determine scale", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save output
        out_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(out_path, frame)
        print(f"Processed: {filename} -> Saved to {OUTPUT_DIR}/")

    print("\n[SUCCESS] Finished processing all photos!")
    print(f"You can review the measurements inside the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    process_offline_frames()
