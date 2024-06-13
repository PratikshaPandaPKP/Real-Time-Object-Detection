from typing import List, Any
import cv2

thres = 0.45  # Threshold to detect objects

# Access the correct camera index (adjust if needed)
cap = cv2.VideoCapture(0)

# Set camera resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 70)  # Adjust brightness if needed

classNames = []
with open("coco.names", "rt") as f:  # Use double quotes for file path
    classNames = f.read().rstrip("\n").split("\n")  # Directly assign to classNames

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()

    if success:  # Only proceed if a frame is read successfully
        classIds, confs, bbox = net.detect(img, confThreshold=thres)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(
                    img,
                    classNames[classId - 1].upper(),
                    (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    img,
                    str(round(confidence * 100, 2)) + "%",  # Add percentage symbol
                    (box[0] + 200, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

    cv2.imshow("Output", img)
    if cv2.waitKey(1) == ord("q"):  # Exit on 'q' key press
        break

cap.release()
cv2.destroyAllWindows()