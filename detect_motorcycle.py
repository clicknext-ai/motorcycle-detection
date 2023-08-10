from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator, colors
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

def draw_boxes(frame, boxes):
    """Draw detected bounding boxes on image frame"""

    # Create annotator object
    annotator = Annotator(frame)
    for box in boxes:
        class_id = box.cls
        class_name = model.names[int(class_id)]
        coordinator = box.xyxy[0]
        confidence = box.conf

        # Draw bounding box
        annotator.box_label(box=coordinator, label=class_name, color=colors(class_id, True))
            
    return annotator.result()

def detect_motorcycle(frame):
    """ Detect motorcycle from image frame """
    
    # Detect motorcycle from image frame
    # All classes: https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
    consider_classes = [0, 3] # 0: person, 3: motorcycle
    confidence_threshold = 0.5
    results = model.predict(frame, conf=confidence_threshold, classes=consider_classes)

    for result in results:
        frame = draw_boxes(frame, result.boxes)

    return frame

if __name__ == "__main__":

    video_path = "motorcycle_video.mov"
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    video_writer = cv2.VideoWriter(video_path + "_demo.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, (1920, 1080))

    while cap.isOpened():

        # Read image frame
        ret, frame = cap.read()

        if ret:

            # Detect motorcycle from image frame
            frame_result = detect_motorcycle(frame)

            # Write result to video
            video_writer.write(frame_result)

            # Show result
            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            cv2.imshow("Video", frame_result)
            cv2.waitKey(30)

        else:
            break


    # Release the VideoCapture object and close the window
    cap.release()
    cv2.destroyAllWindows()