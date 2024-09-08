import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Streamlit UI
st.title("Maritime Object Detection")

# Model selection
model_choice = st.selectbox(
    "Choose a detection model:",
    ["Light Detection Model", "YOLO Small Object", "YOLO Low Light Enhancement"]
)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Convert the image to RGB (for Streamlit display)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    if model_choice == "Light Detection Model":
        # Run the custom light detection
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define HSV ranges for red, green, and white colors
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])

        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 25, 255])

        # Create masks for red, green, and white colors
        red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        red_mask = cv2.add(red_mask1, red_mask2)

        green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
        white_mask = cv2.inRange(hsv_img, lower_white, upper_white)

        # Function to draw bounding boxes and calculate centers
        def draw_bbox_and_center(contours, color, img, bbox_thickness=8):
            centers = []
            for contour in contours:
                if cv2.contourArea(contour) > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, bbox_thickness)
                    center_x, center_y = x + w // 2, y + h // 2
                    centers.append((center_x, center_y))
                    cv2.circle(img, (center_x, center_y), 5, color, -1)
            return centers

        # Find and draw contours for red, green, and white areas
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        red_centers = draw_bbox_and_center(red_contours, (0, 0, 255), img)

        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        green_centers = draw_bbox_and_center(green_contours, (0, 255, 0), img)

        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        white_centers = draw_bbox_and_center(white_contours, (255, 255, 255), img)

        all_centers = red_centers + green_centers + white_centers

        # Draw lines between centers
        if len(all_centers) > 1:
            for i in range(len(all_centers)):
                for j in range(i + 1, len(all_centers)):
                    cv2.line(img, all_centers[i], all_centers[j], (255, 0, 0), 2)

        # Classify lights' positions (left, right, top, bottom)
        if len(all_centers) > 1:
            avg_x = np.mean([x for x, y in all_centers])
            avg_y = np.mean([y for x, y in all_centers])

            for center in all_centers:
                x, y = center
                position = []
                if x < avg_x:
                    position.append("Left")
                else:
                    position.append("Right")

                if y < avg_y:
                    position.append("Top")
                else:
                    position.append("Bottom")

                st.write(f"Light at ({x}, {y}) is on the {' and '.join(position)}")

        # Convert the processed image back to RGB and display
        img_with_boxes = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_with_boxes, caption="Detected Lights", use_column_width=True)

    else:
        # YOLO inference for YOLO-based models
        st.write(f"Running inference with {model_choice}")

        # Load the appropriate YOLO model
        if model_choice == "YOLO Small Object":
            model_path = "path_to_your_yolo_small_object_weights.pt"
        elif model_choice == "YOLO Low Light Enhancement":
            model_path = "path_to_your_yolo_low_light_weights.pt"

        # Load the YOLO model
        model = YOLO(model_path)

        # Run YOLO inference
        results = model(img_rgb)

        # Visualize the results with bounding boxes
        for result in results:
            img_with_boxes = result.plot()

        # Convert the BGR image to RGB for Streamlit display
        img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        st.image(img_with_boxes_rgb, caption=f"Detected Objects with {model_choice}", use_column_width=True)
