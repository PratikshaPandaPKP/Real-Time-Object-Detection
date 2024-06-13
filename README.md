1. **Imports and Setup:**
   - Imports necessary libraries: `cv2` for computer vision operations.
   - Defines `thres` as the confidence threshold to detect objects.
   - Opens the default camera (index 0) using `cv2.VideoCapture`.
   - Sets optional camera properties like frame width, height, and brightness.

2. **Loading Class Names:**
   - Reads class names from a file named "coco.names" containing labels for objects that the model can detect (e.g., person, car, chair).

3. **Loading the Model:**
   - Defines paths to the model configuration file (`configPath`) and the model weights file (`weightsPath`).
   - Initializes the model using `cv2.dnn_DetectionModel` with parameters for input size, scale, mean subtraction, and color swapping.

4. **Main Loop (`while True`):**
   - Continuously reads frames from the webcam (`cap.read()`).
   - Checks if a frame is successfully read (`success` flag).
   - If successful, detects objects in the frame using `net.detect`, which returns class IDs, confidence scores (`confs`), and bounding box coordinates (`bbox`) for detected objects.

5. **Object Detection Visualization:**
   - Iterates through the detected objects (`classIds`, `confs`, `bbox`).
   - Draws a rectangle around each detected object using `cv2.rectangle`.
   - Displays the object class name (`classNames[classId - 1]`) and confidence score as text using `cv2.putText`.

6. **Displaying the Output:**
   - Shows the annotated frame with detected objects in a window titled "Output" using `cv2.imshow`.
   - The loop continues until the user presses 'q', which triggers the `cv2.waitKey(1)` condition to break out of the loop.

7. **Cleanup:**
   - Releases the camera (`cap.release()`) and closes all OpenCV windows (`cv2.destroyAllWindows()`) after exiting the loop.

**Purpose:**
This script demonstrates how to use a pre-trained SSD model from the COCO dataset for real-time object detection on a webcam feed. It showcases basic functionalities such as initializing the model, processing frames, detecting objects, and visualizing the results using OpenCV's drawing and text functions.

**Notes:**
Ensure that the paths to "coco.names", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt", and "frozen_inference_graph.pb" are correctly specified and accessible. Adjust `thres` (threshold) and camera properties (`cap.set()`) as needed for different environments or requirements.

**Video link of the project in action:** 
https://www.loom.com/share/c4764f81d0d64357ab3a9bbddfc1981d?sid=3ce5d01f-49bd-455e-bb67-d3c75a5c9be3
