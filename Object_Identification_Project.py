import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading

class ObjectDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection")
        self.window_width = 500
        self.window_height = 314

        # Load background image
        self.load_background_image()

        # Create frame to hold buttons
        self.button_frame = tk.Frame(self.root, bg="white")
        self.button_frame.pack(pady=10)

        # Create buttons for different functionalities
        self.image_button = tk.Button(self.button_frame, text="Detect Objects in Image", command=self.detect_objects_in_image)
        self.image_button.pack(side="left", padx=10)

        self.video_button = tk.Button(self.button_frame, text="Detect Objects in Video", command=self.detect_objects_in_video)
        self.video_button.pack(side="left", padx=10)

        self.webcam_button = tk.Button(self.button_frame, text="Detect Objects in Webcam", command=self.detect_objects_in_webcam)
        self.webcam_button.pack(side="left", padx=10)

        # Initialize variables
        self.net, self.output_layers, self.classes = self.load_yolov3_model()

    def load_background_image(self):
        # Load background image
        img = Image.open("yolologo2.png")

        # Resize image to fit the window
        img = img.resize((self.window_width, self.window_height), Image.LANCZOS)

        # Convert image to PhotoImage
        self.background_image = ImageTk.PhotoImage(img)

        # Create canvas to display background image
        self.canvas = tk.Canvas(self.root, width=self.window_width, height=self.window_height)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.background_image)

    def load_yolov3_model(self):
        # Load YOLOv3 model and classes
        net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        return net, output_layers, classes

    def preprocess_image(self, image_path):
        # Preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        return img, blob

    def perform_detection(self, frame):
        # Perform object detection
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return frame

    def detect_objects_in_image(self):
        # Detect objects in an image
        image_path = filedialog.askopenfilename()
        if image_path:
            img, _ = self.preprocess_image(image_path)
            frame = self.perform_detection(img)
            cv2.imshow("Object Detection", cv2.resize(frame, (1280, 780)))  # Adjust window size here
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def detect_objects_in_video(self):
        # Detect objects in a video
        video_path = filedialog.askopenfilename()
        if video_path:
            video_capture = cv2.VideoCapture(video_path)
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break
                frame = self.perform_detection(frame)
                cv2.imshow("Object Detection", cv2.resize(frame, (1280, 780)))  # Adjust window size here
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            video_capture.release()
            cv2.destroyAllWindows()

    def detect_objects_in_webcam(self):
        # Detect objects in webcam feed
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            frame = self.perform_detection(frame)
            cv2.imshow("Object Detection", cv2.resize(frame, (1280, 780)))  # Adjust window size here
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionGUI(root)
    root.mainloop()
