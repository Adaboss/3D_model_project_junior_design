import cv2
import numpy as np
import os
from pathlib import Path

def get_blue_mask(img):
    """Returns a binary mask of blue pixels in the image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define standard HSV range for blue color
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    return mask

def find_global_object_bounds(image_files):
    """Finds the min X, min Y, max X, max Y defining the global bounding box for the blue object across all frames."""
    global_min_x = float('inf')
    global_min_y = float('inf')
    global_max_x = -float('inf')
    global_max_y = -float('inf')
    
    first_img_shape = None
    
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is None:
            continue
            
        if first_img_shape is None:
            first_img_shape = img.shape
            
        mask = get_blue_mask(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Only consider contours with a reasonable minimum area to avoid noise
            if cv2.contourArea(largest_contour) > 100:
                x, y, w, h = cv2.boundingRect(largest_contour)
                global_min_x = min(global_min_x, x)
                global_min_y = min(global_min_y, y)
                global_max_x = max(global_max_x, x + w)
                global_max_y = max(global_max_y, y + h)
                
    if global_min_x == float('inf'):
        return None, first_img_shape
    return (global_min_x, global_min_y, global_max_x, global_max_y), first_img_shape

def process_images_and_reconstruct(input_dir="frames", output_cropped_dir="cropped_frames", reconstruction_dir="reconstruction_output"):
    input_path = Path(input_dir)
    cropped_path = Path(output_cropped_dir)
    recon_path = Path(reconstruction_dir)
    
    try:
        import pycolmap
    except ImportError:
        print("Error: pycolmap is not installed/found.")
        print("Please securely install it using: pip install pycolmap")
        print("The script will still run to crop the images for you.")
        pycolmap = None

    if not input_path.exists():
        print(f"Input directory '{input_dir}' not found. Please ensure it exists and has images.")
        return
        
    cropped_path.mkdir(parents=True, exist_ok=True)
    recon_path.mkdir(parents=True, exist_ok=True)

    # Collect .jpg and .png images
    image_files = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")))
    
    if not image_files:
        print(f"No images found in '{input_dir}'")
        return

    print("Step 1/3: Analyzing frames to find global blue object bounds...")
    bounds, img_shape = find_global_object_bounds(image_files)
    
    if bounds is None:
        print("No blue object found in any frame. Cannot crop specifically to the target object.")
        return
        
    g_min_x, g_min_y, g_max_x, g_max_y = bounds
    w = g_max_x - g_min_x
    h = g_max_y - g_min_y
    
    print(f"Global object bounds found: x=[{g_min_x}, {g_max_x}], y=[{g_min_y}, {g_max_y}], w={w}, h={h}")
    
    # We apply a consistent padding around the bounding box to capture enough 
    # surrounding environment/edges for the module to recognize structure in the model cleanly.
    # 40% of the maximum dimension or at least 100 pixels.
    padding = max(int(max(w, h) * 0.4), 100) 
    
    # Ensure standard box within image bounds
    crop_x = max(0, g_min_x - padding)
    crop_y = max(0, g_min_y - padding)
    crop_x2 = min(img_shape[1], g_max_x + padding)
    crop_y2 = min(img_shape[0], g_max_y + padding)
    
    print(f"Padded global crop window applied: [{crop_x}:{crop_x2}, {crop_y}:{crop_y2}]")
    
    print("Step 2/3: Cropping images uniformly and saving to output directory...")
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is None:
            continue
            
        cropped_img = img[crop_y:crop_y2, crop_x:crop_x2]
        output_file = cropped_path / img_file.name
        cv2.imwrite(str(output_file), cropped_img)
        
    print(f"Successfully cropped {len(image_files)} images into '{output_cropped_dir}'.")
    
    if pycolmap is None:
        print("Exiting because pycolmap is unavailable. Images are cropped and ready!")
        return
        
    print("Step 3/3: Running pyCOLMAP 3D Reconstruction...")
    database_path = recon_path / "database.db"
    
    # Starting fresh so old features aren't carried over a new script run
    if database_path.exists():
        database_path.unlink() 
        
    print("Extracting SIFT features (using CUDA GPU acceleration)...")
    pycolmap.extract_features(
        database_path=database_path, 
        image_path=cropped_path,
        sift_options={'use_gpu': True}
    )
    
    print("Matching features exhaustively (using CUDA GPU acceleration)...")
    pycolmap.match_exhaustive(
        database_path=database_path,
        sift_options={'use_gpu': True}
    )
    
    print("Incremental mapping (Reconstructing 3D structure)...")
    # This process will automatically read the EXIF data or construct camera dimensions based on identical crop size
    maps = pycolmap.incremental_mapping(
        database_path=database_path, 
        image_path=cropped_path, 
        output_path=recon_path
    )
    
    print("Reconstruction pipeline finished!")
    if maps:
        print(f"Success! Models and configurations are saved in: {reconstruction_dir}")
    else:
        print("Warning: Reconstruction step resulted in an empty sparse map.")

if __name__ == "__main__":
    process_images_and_reconstruct()
