import cv2
import numpy as np
import os

def main():
    """
    Runs Task 1 and Task 2 on a single image file.
    """
    
    # 1. READ A SINGLE IMAGE (Modification for static image processing)
    # -------------------------------------------------------------------
    image_path = "path/to/your/object_image.jpg"  # <-- **CHANGE THIS PATH**
    
    # Read the image in BGR format
    frame = cv2.imread("cup.png")
    
    if frame is None:
        print(f"Error: Could not read image from {image_path}. Check the path.")
        return
        
    # ==========================================================
    # TASK 1: Basic Image I/O and Color Spaces
    # ==========================================================
    
    # [cite_start]Convert the frame from BGR (OpenCV default) to HSV [cite: 1]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # [cite_start]Display the required frames [cite: 1]
    cv2.imshow("Raw Frame (Task 1)", frame)
    cv2.imshow("HSV Frame (Task 1)", hsv)
    
    # [cite_start]Save the images to disk (Task 1 Deliverable) [cite: 1]
    cv2.imwrite("original.png", frame)
    cv2.imwrite("hsv.png", hsv)
    print("Saved original.png and hsv.png to disk.")
    
    # ==========================================================
    # TASK 2: Color-Based Segmentation
    # ==========================================================
    
    # [cite_start]Define lower and upper bounds for the target color (Example: Red) in HSV [cite: 1]
    # **NOTE: You will need to tune these values for your specific object and lighting**
    
    #mask value 
    lower1 = np.array([85, 80, 30])
    upper1 = np.array([115, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1)

    
    # lower2 = np.array([170, 120, 70])
    # upper2 = np.array([180, 255, 255])
    # mask2 = cv2.inRange(hsv, lower2, upper2)
    
    # mask = mask1 | mask2 
    
    # [cite_start]Visualize the binary mask [cite: 1]
    cv2.imshow("Binary Mask (Task 2)", mask)
    
    # [cite_start]Apply the mask to the original image to show only the segmented object [cite: 1]
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    # [cite_start]Visualize the segmented result [cite: 1]
    cv2.imshow("Segmented Object (Task 2)", result)

    # ==========================================================
    # TASK 3: Noise Reduction and Contour Detection
    # ==========================================================

    # 1. Clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Clean Mask (Task 3)", mask_clean)

    # 2. Find contours
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx, cy = None, None

    if contours:
        # 3. Largest contour
        largest = max(contours, key=cv2.contourArea)

        # 4. Compute centroid
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 5. Draw contour + centroid on the frame
            cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

    cv2.imshow("Tracked Object (Task 3)", frame)

    # PRINT centroid for your report
    print("Centroid:", cx, cy)





    # ==========================================================
    # Loop Control (Modified for single image)
    # ==========================================================
    
    # Wait indefinitely until any key is pressed, then close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ensure all required libraries are installed:
    # pip install opencv-python numpy
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")

