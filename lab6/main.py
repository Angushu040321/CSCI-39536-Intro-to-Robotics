import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    """
    Runs Task 1 and Task 2 on a single image file.
    """
    
    # 1. READ A SINGLE IMAGE (Modification for static image processing)
    # -------------------------------------------------------------------
    #image_path = "path/to/your/object_image.jpg"  # <-- **CHANGE THIS PATH**
    
    # Read the image in BGR format
    frame = cv2.imread("images/cup.png")
    
    #if frame is None:
    #    print(f"Error: Could not read image from {image_path}. Check the path.")
     #   return
        
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

    #======================================================
    #TASK 4:
    #======================================================
    # ----------------------------
    # Load the SMALL video you created earlier
    # ----------------------------
    video_path = "images\mini_ball.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open mini_ball.mp4")
        exit()

    cx_history = []
    frames_for_video = []

    # HSV threshold for GREEN tennis ball
    lower_green = np.array([35, 80, 80])
    upper_green = np.array([85, 255, 255])

    frame_count = 0
    max_frames = 300  # ~10 seconds of video

    # ----------------------------
    # Prepare video writer for tracked output
    # ----------------------------
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter('tracked_output.mp4', fourcc, 20.0, (640, 360))

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Frame is already 640x360 if you used shrink_video.py
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Apply HSV threshold
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Morphological clean up
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx = np.nan
        command = "SEARCH"

        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw contour + centroid
                cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                # ----------------------------
                # CONTROL LOGIC (Task 4)
                # ----------------------------
                h, w, _ = frame.shape
                center_x = w // 2

                if cx < center_x - w * 0.1:
                    command = "TURN_LEFT"
                elif cx > center_x + w * 0.1:
                    command = "TURN_RIGHT"
                else:
                    command = "FORWARD"

        # Draw command on video frame
        cv2.putText(frame, f"Command: {command}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Save centroid and frame
        cx_history.append(cx)
        out_vid.write(frame)

        frame_count += 1

    cap.release()
    out_vid.release()
    print("Saved tracked video as tracked_output.mp4")

    # ----------------------------
    # Save cx plot
    # ----------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(cx_history, color='blue')
    plt.title("Object Centroid X Over Time")
    plt.xlabel("Frame")
    plt.ylabel("cx (pixels)")
    plt.tight_layout()
    plt.savefig("cx_plot.png")

    print("Saved plot as cx_plot.png")

    # =============================================================
    # TASK 5: Testing and Robustness
    # =============================================================

    print("\nTask 5: Testing Different Conditions")

    # test different lighting
    tests = [
        {"name": "normal", "contrast": 1.0, "brightness": 0},
        {"name": "bright", "contrast": 1.2, "brightness": 50},
        {"name": "dim", "contrast": 0.8, "brightness": -50},
    ]

    results = []

    for test in tests:
        print(f"Testing: {test['name']}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        detections = 0
        total = 0

        out_name = f"task5_{test['name']}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_name, fourcc, 20.0, (640, 360))

        while total < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # adjust lighting
            frame = cv2.convertScaleAbs(frame, alpha=test["contrast"], beta=test["brightness"])

            if frame.shape[1] != 640 or frame.shape[0] != 360:
                frame = cv2.resize(frame, (640, 360))

            # same detection as before
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_green, upper_green)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            command = "SEARCH"

            if contours:
                biggest = max(contours, key=cv2.contourArea)
                M = cv2.moments(biggest)

                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    cv2.drawContours(frame, [biggest], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                    h, w, _ = frame.shape
                    mid = w // 2

                    if cx < mid - w * 0.1:
                        command = "TURN_LEFT"
                    elif cx > mid + w * 0.1:
                        command = "TURN_RIGHT"
                    else:
                        command = "FORWARD"

                    detections += 1

            cv2.putText(frame, f"Command: {command}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(frame)
            total += 1

        cap.release()
        out.release()

        rate = (detections / total * 100) if total > 0 else 0
        results.append((test['name'], rate, detections, total))
        print(f"  {test['name']}: {rate:.1f}% ({detections}/{total})")

    with open("task5_results.txt", "w") as f:
        f.write("Task 5 Results\n\n")
        for name, rate, detected, total in results:
            f.write(f"{name}: {rate:.1f}% ({detected}/{total})\n")

    # =============================================================
    # TASK 6: MobileNet-SSD vs Color Detection
    # =============================================================

    print("\nTask 6: MobileNet-SSD Detection")

    # use different video of a man for task 6 bc no tennis ball option
    video_path = "images/man_walking.mp4"

    prototxt_path = "models/deploy.prototxt"
    model_path = "models/mobilenet_iter_73000.caffemodel"
    TARGET_CLASS_ID = 15  # person
    CONF_THRESH = 0.5

    CLASS_LABELS = ["background", "aeroplane", "bicycle", "bird", "boat",
                    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                    "sofa", "train", "tvmonitor"]

    print(f"Target class: {CLASS_LABELS[TARGET_CLASS_ID]}")

    try:
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        print("Model loaded")
    except Exception as e:
        print(f"Error: {e}")
        net = None

    if net is not None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video")
        else:
            print("Processing video...")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_vid = cv2.VideoWriter('images/task6_comparison.mp4', fourcc, 20.0, (640, 360))

            color_success = 0
            detector_success = 0
            total_frames = 0
            frame_count = 0

            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame.shape[1] != 640 or frame.shape[0] != 360:
                    frame = cv2.resize(frame, (640, 360))

                frame_display = frame.copy()
                (h, w) = frame.shape[:2]
                center_x = w // 2

                # Color-based detection
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_green, upper_green)
                kernel = np.ones((5, 5), np.uint8)
                mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                cx_color, cy_color = None, None
                command_color = "SEARCH"

                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest)

                    if M["m00"] != 0:
                        cx_color = int(M["m10"] / M["m00"])
                        cy_color = int(M["m01"] / M["m00"])

                        cv2.drawContours(frame_display, [largest], -1, (0, 255, 0), 2)
                        cv2.circle(frame_display, (cx_color, cy_color), 6, (0, 255, 0), -1)
                        cv2.putText(frame_display, "Color", (cx_color - 30, cy_color - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if cx_color < center_x - w * 0.1:
                            command_color = "TURN_LEFT"
                        elif cx_color > center_x + w * 0.1:
                            command_color = "TURN_RIGHT"
                        else:
                            command_color = "FORWARD"

                        color_success += 1

                # MobileNet-SSD detection
                blob = cv2.dnn.blobFromImage(
                    frame,
                    scalefactor=0.007843,
                    size=(300, 300),
                    mean=127.5
                )

                net.setInput(blob)
                detections = net.forward()

                cx_det, cy_det = None, None
                command_det = "SEARCH"

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence < CONF_THRESH:
                        continue

                    class_id = int(detections[0, 0, i, 1])
                    if class_id != TARGET_CLASS_ID:
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    cx_det = (x1 + x2) // 2
                    cy_det = (y1 + y2) // 2

                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(frame_display, (cx_det, cy_det), 5, (255, 0, 0), -1)
                    cv2.putText(frame_display, "Detector", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    if cx_det < center_x - w * 0.1:
                        command_det = "TURN_LEFT"
                    elif cx_det > center_x + w * 0.1:
                        command_det = "TURN_RIGHT"
                    else:
                        command_det = "FORWARD"

                    detector_success += 1
                    break

                cv2.putText(frame_display, f"Color Cmd: {command_color}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.putText(frame_display, f"Det Cmd: {command_det}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                out_vid.write(frame_display)
                total_frames += 1
                frame_count += 1

            cap.release()
            out_vid.release()

            print("Video saved")

            color_rate = (color_success / total_frames * 100) if total_frames > 0 else 0
            detector_rate = (detector_success / total_frames * 100) if total_frames > 0 else 0

            print(f"\nResults:")
            print(f"Color detection: {color_rate:.1f}%")
            print(f"MobileNet-SSD: {detector_rate:.1f}%")

            plt.figure(figsize=(8, 5))
            methods = ['Color-based', 'MobileNet-SSD']
            rates = [color_rate, detector_rate]
            plt.bar(methods, rates, color=['green', 'pink'])
            plt.title("Detection Success Rate")
            plt.ylabel("Success Rate (%)")l
            plt.ylim(0, 100)
            plt.savefig("task6_comparison.png")
            plt.close()

            with open("task6_results.txt", "w") as f:
                f.write("Task 6 Results\n\n")
                f.write(f"Color detection: {color_rate:.1f}%\n")
                f.write(f"MobileNet-SSD: {detector_rate:.1f}%\n\n")
                f.write("Colorbased is faster but sensitive to lighting\n")
                f.write("MobileNet SSD is robuster but slower\n")

if __name__ == "__main__":
    # Ensure all required libraries are installed:
    # pip install opencv-python numpy
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")

