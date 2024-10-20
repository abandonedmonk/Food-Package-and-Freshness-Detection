import os
import io
import base64
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from collections import defaultdict
import cv2
from groq import Groq
import math
from io import BytesIO
from queue import Queue
import time
import threading

# Load YOLO models
# packet_model_path = 'C:/Flipkart_GRID/final/packet_detection_v2.pt'
packet_model_path = 'C:/Flipkart_GRID/Flipkart_v2/packet_detection_v2.pt'
tomato_model_path = 'C:/Flipkart_GRID/Flipkart_v2/Tomato_Model.pt'

tomato_model = YOLO(tomato_model_path)
packet_model = YOLO(packet_model_path)


class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 50:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids


def encode_image_to_base64(cv2_img):
    cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img_rgb)

    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


image_queue = Queue(maxsize=100)
results_queue = Queue()


def ensure_directory_and_file(filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(filename):
        open(filename, 'w').close()


def process_with_llama(timeout=300):
    # client = Groq(
    #     api_key="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")
    client = Groq(
        api_key="")
    start_time = time.time()

    while time.time() - start_time < timeout:
        if not image_queue.empty():
            img_data = image_queue.get()
            cropped_img = img_data['image']
            frame_id = img_data['frame_id']
            object_id = img_data['object_id']

            img_base64 = encode_image_to_base64(cropped_img)

            try:
                prompt = """
                    Extract all relevant details from the food packaging, including but not limited to:
                    - Product name
                    - Brand name
                    - Ingredients list
                    - Nutritional information (calories, fats, proteins, carbohydrates, etc.)
                    - Serving size
                    - Expiry date or 'best before' date
                    - Manufacturing date
                    - Weight/quantity
                    - Storage instructions
                    - Allergen warnings
                    - Certifications (e.g., organic, non-GMO)
                    - Country of origin
                    - Contact information (manufacturer or distributor)
                    - Batch/lot number
                    - Cooking or usage instructions (if any)
                    - Any legal disclaimers (e.g., 'Keep out of reach of children')

                    Ensure that the extracted information is categorized under the appropriate headings for ease of understanding.
                    """

                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"}}
                            ]
                        }
                    ],
                    model="llama-3.2-11b-vision-preview",
                    temperature=0.1,
                    top_p=1,
                )

                response = chat_completion.choices[0].message.content

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"results/packet_{frame_id}_{
                    object_id}_{timestamp}.txt"
                ensure_directory_and_file(filename)

                with open(filename, 'w') as f:
                    f.write(f"Frame ID: {frame_id}\n")
                    f.write(f"Object ID: {object_id}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write("Analysis:\n")
                    f.write(response)
                    f.write("\n-------------------\n")

                results_queue.put({
                    'frame_id': frame_id,
                    'object_id': object_id,
                    'result': response
                })

            except Exception as e:
                print(f"Error processing image {
                      frame_id}_{object_id}: {str(e)}")

            image_queue.task_done()

        time.sleep(0.1)

    # If we've timed out, add a message to the results queue
    if time.time() - start_time >= timeout:
        results_queue.put({
            'frame_id': 'timeout',
            'object_id': 'timeout',
            'result': 'Analysis timed out. Please try again.'
        })


tracker = Tracker()


def count_tomatoes(results):
    b_fully_ripened = 0
    b_half_ripened = 0
    b_green = 0
    l_fully_ripened = 0
    l_half_ripened = 0
    l_green = 0

    for result in results:
        for cls in result.boxes.cls:
            if cls == 0:  # 'b_fully_ripened'
                b_fully_ripened += 1
            elif cls == 1:  # 'b_half_ripened'
                b_half_ripened += 1
            elif cls == 2:  # 'b_green'
                b_green += 1
            elif cls == 3:  # 'l_fully_ripened'
                l_fully_ripened += 1
            elif cls == 4:  # 'l_half_ripened'
                l_half_ripened += 1
            elif cls == 5:  # 'l_green'
                l_green += 1

    fresh = b_fully_ripened + l_fully_ripened
    not_fresh = b_half_ripened + b_green + l_half_ripened + l_green

    return {
        'b_fully_ripened': b_fully_ripened,
        'b_half_ripened': b_half_ripened,
        'b_green': b_green,
        'l_fully_ripened': l_fully_ripened,
        'l_half_ripened': l_half_ripened,
        'l_green': l_green,
        'fresh': fresh,
        'not_fresh': not_fresh
    }


def process_frame(frame, model_type):
    if model_type == "tomato":
        model = tomato_model
    else:
        model = packet_model

    results = model(frame)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = result.names[int(box.cls[0])]
            detections.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

    tracked_objects = tracker.update(detections)

    if model_type == "packet":
        for obj in tracked_objects:
            x, y, w, h, object_id = obj
            cropped_img = frame[y:y+h, x:x+w]
            image_queue.put({
                'image': cropped_img,
                'frame_id': time.time(),
                'object_id': object_id
            })

    if model_type == "tomato":
        tomato_counts = count_tomatoes(results)
    else:
        tomato_counts = None

    return tracked_objects, results, tomato_counts


def process_live_stream(model_type):
    FRAME_WINDOW = st.image([])
    ANALYSIS_OUTPUT = st.empty()
    COUNTER_OUTPUT = st.empty()

    start_detection = st.button("Start Detection", key="start_button")
    stop_detection = st.button("Stop Detection", key="stop_button")

    if start_detection:
        cap = cv2.VideoCapture(0)
        frame_count = 0
        down = {}  # Dictionary to track objects that have crossed the line
        counter_down = set()  # Set to store unique object IDs
        latest_result = ""

        if model_type == "packet":
            llama_thread = threading.Thread(target=process_with_llama)
            llama_thread.daemon = True
            llama_thread.start()

        stop_flag = False

        while True:
            if stop_detection:
                stop_flag = True
                st.warning("Detection stopped.")
                break

            ret, frame = cap.read()
            if not ret or stop_flag:
                break

            frame = cv2.resize(frame, (640, 480))
            original_frame = frame.copy()

            if model_type == "packet":
                results = packet_model(frame, stream=True)
                list_bbox = []

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        confidence = float(box.conf[0])

                        if confidence >= 0.5:
                            list_bbox.append([x1, y1, x2, y2])
                            cv2.rectangle(frame, (x1, y1),
                                          (x2, y2), (0, 255, 0), 2)

                # Update tracker with detected boxes
                bbox_id = tracker.update(list_bbox)
                y_line = 240  # Horizontal line position
                offset = 5

                # Draw tracking line
                cv2.line(frame, (0, y_line), (640, y_line), (0, 0, 255), 2)

                for bbox in bbox_id:
                    x3, y3, x4, y4, id = bbox
                    cx = (x3 + x4) // 2
                    cy = (y3 + y4) // 2

                    # Check if object crosses the line
                    if abs(cy - y_line) < offset and id not in down:
                        down[id] = cy
                        counter_down.add(id)

                        # Crop and process the packet image
                        cropped_img = original_frame[int(
                            y3):int(y4), int(x3):int(x4)]
                        if cropped_img.size > 0:
                            image_queue.put({
                                'image': cropped_img,
                                'frame_id': frame_count,
                                'object_id': id
                            })

                # Display counter
                cv2.putText(frame, f'Packets Detected: {len(counter_down)}',
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                COUNTER_OUTPUT.text(f'Total Packets Detected: {
                                    len(counter_down)}')

                # Check for Llama results
                while not results_queue.empty():
                    result = results_queue.get()
                    if result['object_id'] != 'timeout':
                        latest_result = f"Analysis for Packet {
                            result['object_id']}:\n{result['result']}"
                        ANALYSIS_OUTPUT.text_area(
                            "Latest Analysis", latest_result, height=300)

            else:  # Tomato detection
                results = tomato_model(frame, stream=True)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        confidence = float(box.conf[0])
                        cls = int(box.cls[0])

                        if confidence >= 0.5:
                            class_name = tomato_model.names[cls]
                            cv2.rectangle(frame, (x1, y1),
                                          (x2, y2), (0, 255, 0), 2)
                            label = f'{class_name} {confidence:.2f}'
                            cv2.putText(frame, label, (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                tomato_counts = count_tomatoes([r])
                if tomato_counts:
                    report = f"""
                    Big Tomatoes:
                    Fully Ripened: {tomato_counts['b_fully_ripened']}
                    Half Ripened: {tomato_counts['b_half_ripened']}
                    Green: {tomato_counts['b_green']}
                    Cherry Tomatoes:
                    Fully Ripened: {tomato_counts['l_fully_ripened']}
                    Half Ripened: {tomato_counts['l_half_ripened']}
                    Green: {tomato_counts['l_green']}
                    Fresh Tomatoes: {tomato_counts['fresh']}
                    Not Fresh Tomatoes: {tomato_counts['not_fresh']}
                    """
                    ANALYSIS_OUTPUT.text(report)

            # Convert BGR to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)
            frame_count += 1
            time.sleep(0.1)

        cap.release()
        FRAME_WINDOW.image([])


def process_video(uploaded_file, model_type):
    # Save uploaded video temporarily
    temp_file = f"temp_video_{int(time.time())}.mp4"
    with open(temp_file, 'wb') as f:
        f.write(uploaded_file.read())

    # Video processing parameters
    cap = cv2.VideoCapture(temp_file)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create video writer
    output_path = f"annotated_video_{int(time.time())}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (frame_width, frame_height))

    # Progress tracking
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    status_text = st.empty()
    video_placeholder = st.empty()
    analysis_text = st.empty()

    # Initialize tracking variables
    down = {}
    counter_down = set()
    current_frame = 0
    y_line = frame_height // 2
    offset = 5

    if model_type == "packet":
        llama_thread = threading.Thread(target=process_with_llama)
        llama_thread.daemon = True
        llama_thread.start()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_frame += 1
            progress = int(current_frame * 100 / frame_count)
            progress_bar.progress(progress)
            status_text.text(f'Processing frame {
                             current_frame} of {frame_count}')

            if model_type == "packet":
                # Packet detection and tracking logic
                results = packet_model(frame, stream=True)
                list_bbox = []
                annotated_frame = frame.copy()

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        confidence = float(box.conf[0])

                        if confidence >= 0.5:
                            list_bbox.append([x1, y1, x2, y2])
                            cv2.rectangle(annotated_frame, (x1, y1),
                                          (x2, y2), (0, 255, 0), 2)
                            label = f'Packet {confidence:.2f}'
                            cv2.putText(annotated_frame, label, (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                bbox_id = tracker.update(list_bbox)
                cv2.line(annotated_frame, (0, y_line),
                         (frame_width, y_line), (0, 0, 255), 2)

                for bbox in bbox_id:
                    x3, y3, x4, y4, id = bbox
                    cx = (x3 + x4) // 2
                    cy = (y3 + y4) // 2

                    cv2.putText(annotated_frame, f"ID: {id}", (x3, y3-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    if abs(cy - y_line) < offset and id not in down:
                        down[id] = cy
                        counter_down.add(id)

                        cropped_img = frame[int(y3):int(y4), int(x3):int(x4)]
                        if cropped_img.size > 0:
                            image_queue.put({
                                'image': cropped_img,
                                'frame_id': current_frame,
                                'object_id': id
                            })

                cv2.putText(annotated_frame, f'Packets Detected: {len(counter_down)}',
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Process Llama results
                while not results_queue.empty():
                    result = results_queue.get()
                    if result['object_id'] != 'timeout':
                        analysis_text.text_area(f"Analysis for Packet {result['object_id']}",
                                                result['result'], height=150)

            else:  # Tomato detection
                results = tomato_model(frame, stream=True)
                annotated_frame = frame.copy()

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        confidence = float(box.conf[0])
                        cls = int(box.cls[0])

                        if confidence >= 0.5:
                            class_name = tomato_model.names[cls]
                            cv2.rectangle(annotated_frame, (x1, y1),
                                          (x2, y2), (0, 255, 0), 2)
                            label = f'{class_name} {confidence:.2f}'
                            cv2.putText(annotated_frame, label, (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write and display frame
            out.write(annotated_frame)
            if current_frame % 3 == 0:  # Update display every 3 frames
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB")

    finally:
        # Cleanup
        cap.release()
        out.release()
        os.remove(temp_file)

        # Provide download link
        with open(output_path, 'rb') as f:
            st.download_button(
                label="Download Annotated Video",
                data=f,
                file_name="annotated_video.mp4",
                mime="video/mp4"
            )
        os.remove(output_path)


def main():
    st.title("Tomato and Packet Detection with Ripeness Analysis")

    mode = st.sidebar.selectbox(
        "Select Mode", ["Upload Image", "Upload Video", "Live Feed"])
    model_type = st.sidebar.selectbox(
        "Select Model Type", ["tomato", "packet"])

    if mode == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            img_array = np.array(image)
            detections, results, tomato_counts = process_frame(
                img_array, model_type)

            # Display the annotated image
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                st.image(im, caption='Processed Image', use_column_width=True)

            if model_type == "tomato" and tomato_counts:
                st.subheader("Tomato Ripeness Report:")
                st.write(f"Big Tomatoes:")
                st.write(f"  Fully Ripened: {
                         tomato_counts['b_fully_ripened']}")
                st.write(f"  Half Ripened: {tomato_counts['b_half_ripened']}")
                st.write(f"  Green: {tomato_counts['b_green']}")
                st.write(f"Cherry Tomatoes:")
                st.write(f"  Fully Ripened: {
                         tomato_counts['l_fully_ripened']}")
                st.write(f"  Half Ripened: {tomato_counts['l_half_ripened']}")
                st.write(f"  Green: {tomato_counts['l_green']}")
                st.write(f"Fresh Tomatoes: {tomato_counts['fresh']}")
                st.write(f"Not Fresh Tomatoes: {tomato_counts['not_fresh']}")

            elif model_type == "packet":
                with st.spinner('Analyzing detected objects...'):
                    llama_thread = threading.Thread(target=process_with_llama)
                    llama_thread.daemon = True
                    llama_thread.start()

                    # Wait for the Llama thread to finish or timeout
                    llama_thread.join(timeout=15)

                results = []
                while not results_queue.empty():
                    results.append(results_queue.get())

                if results:
                    for result in results:
                        if result['object_id'] == 'timeout':
                            st.warning(result['result'])
                        else:
                            st.subheader(f"Analysis for Object ID: {
                                         result['object_id']}")
                            st.text_area("Llama Analysis",
                                         result['result'], height=300)
                else:
                    st.warning(
                        "No analysis results available. The process might have timed out.")

    elif mode == "Upload Video":
        uploaded_file = st.file_uploader(
            "Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            process_video(uploaded_file, model_type)

    elif mode == "Live Feed":
        process_live_stream(model_type)


if __name__ == "__main__":
    main()
