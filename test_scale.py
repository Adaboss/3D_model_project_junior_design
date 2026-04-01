import cv2
import numpy as np
gray = cv2.imread("frames/capture_001.jpg", 0)
display = cv2.imread("frames/capture_001.jpg")

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

square_sizes = []
for c in cnts:
    area = cv2.contourArea(c)
    if 200 < area < 20000:
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect
        if w > 0 and h > 0:
            aspect = w / float(h)
            if 0.7 <= aspect <= 1.3:
                extent = area / (w * h)
                if extent > 0.75:
                    square_sizes.append((w+h)/2.0)

print("Squares found:", len(square_sizes))
if len(square_sizes) >= 3:
    ppm = np.median(square_sizes) / 8.0
    print("Scale:", ppm)
else:
    print("Failed to find scale")
    import sys; sys.exit(1)

edged = cv2.Canny(blurred, 30, 100)
edged = cv2.dilate(edged, None, iterations=5)
edged = cv2.erode(edged, None, iterations=5)
ocnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("Objects found:", len(ocnts))
for c in ocnts:
    if cv2.contourArea(c) > 1000:
        r = cv2.minAreaRect(c)
        (cx,cy), (w,h), angle = r
        w_mm = w / ppm
        h_mm = h / ppm
        if not ((6 < w_mm < 10) and (6 < h_mm < 10)):
            print(f"Object: {w_mm:.1f}x{h_mm:.1f} mm at ({cx:.1f}, {cy:.1f})")

