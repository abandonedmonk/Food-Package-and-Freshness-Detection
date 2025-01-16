# Tomato and Packet Detection System

![Banner Image](https://i.ibb.co/s325zjv/2ec3edaf-35d4-4cef-a574-a7ccd1e92c5a.jpg)  

A computer vision system that performs real-time detection and analysis of tomatoes and product packets using YOLO object detection and LLaMA vision-language model.

## Features

### Tomato Analysis
- Real-time detection of tomatoes in different ripeness stages
- Classification of tomatoes into categories:
  - Big Tomatoes (Fully Ripened, Half Ripened, Green)
  - Cherry Tomatoes (Fully Ripened, Half Ripened, Green)
- Automatic counting of fresh vs. not fresh tomatoes
- Support for live video feed, uploaded videos, and images

### Packet Detection
- Real-time detection and tracking of product packets
- Detailed packet analysis using LLaMA vision model, including:
  - Product name and brand
  - Ingredients list
  - Nutritional information
  - Expiry dates
  - Storage instructions
  - Allergen warnings
  - Other packaging details
- Automatic counting of detected packets
- Results storage in text files

## Requirements

- Python 3.8+
- OpenCV
- Streamlit
- Ultralytics YOLO
- Pillow
- NumPy
- Groq API access
- CUDA-compatible GPU (recommended for optimal

## Installation

1. Clone the repository
2. Install required packages:
   ```bash
    pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Add your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

## Model Paths

Update the following paths in the code to point to your YOLO model files:
```python
packet_model_path = 'path/to/packet_detection_v2.pt'
tomato_model_path = 'path/to/Tomato_Model.pt'
```

## Usage

Run the application using Streamlit:
```bash
streamlit run final_stream.py
```

### Operating Modes

1. **Upload Image**
   - Upload single images for analysis
   - View detailed detection results and analysis

2. **Upload Video**
   - Process pre-recorded videos
   - Download annotated results
   - View real-time analysis during processing

3. **Live Feed**
   - Real-time detection using webcam
   - Continuous analysis and tracking
   - Live counter for detections

## Output

- Annotated images/videos with bounding boxes
- Detailed analysis reports
- Counting statistics
- Text files with packet analysis results (stored in `results` directory)

## Technical Details

- Uses YOLO (You Only Look Once) for object detection
- Implements object tracking using center points method
- Integrates LLaMA vision model for detailed packet analysis
- Multi-threaded processing for simultaneous detection and analysis
- Results queuing system for asynchronous processing

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


https://youtu.be/oQ-eejeq8PQ
[![Video Title](https://img.youtube.com/vi/oQ-eejeq8PQ/0.jpg)](https://www.youtube.com/watch?v=oQ-eejeq8PQ)
