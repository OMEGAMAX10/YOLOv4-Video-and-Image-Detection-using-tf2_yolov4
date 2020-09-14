import os
import cv2
import tensorflow as tf
from pathlib import Path
from numpy import loadtxt
from keras.utils import get_file
from tf2_yolov4.model import YOLOv4
from tf2_yolov4.anchors import YOLOV4_ANCHORS

class_list = [cls.strip() for cls in open("yolov4_utils/coco_classes.txt")]  # COCO classes
color_list = loadtxt("yolov4_utils/colors.txt").tolist()  # box colors
if os.path.isdir("images") is False:
    os.mkdir("images")
if os.path.isdir("videos") is False:
    os.mkdir("videos")


def get_processed_image(img, boxes, scores, classes):
    """
        Draws the class boxes with their scores on a OpenCV image

        Arguments:
            img -- image opened with OpenCV
            boxes -- the list of boxes(their coordinates)
            scores -- the list of scores for each box
            classes -- the list of classes for each box

        Returns:
            img -- processed image with all the boxes drawn
    """
    for (xmin, ymin, xmax, ymax), score, cl in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
        if score > 0:
            start_point = (int(xmin), int(ymin))
            end_point = (int(xmax), int(ymax))
            color = color_list[cl % 6]
            img = cv2.rectangle(img, start_point, end_point, color, 2)  # draw class box
            text = f'{class_list[cl]}: {score:0.2f}'
            (test_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_ITALIC, 0.5, 1)
            end_point = (int(xmin) + test_width + 2, int(ymin) - text_height - 2)
            img = cv2.rectangle(img, start_point, end_point, (0, 255, 255), -1)
            cv2.putText(img, text, start_point, cv2.FONT_ITALIC, 0.5, 0, 1)  # print class type with score
    return img


def detect_image(image_path, output_path="output_images"):
    """
        Detects the objects from an image using a YOLOv4 model and returns the result image with the same name

        Arguments:
            image_path -- path of the image to run detection on
            output_path -- the path of the output folder

        Returns:
            output_image_path -- path to processed image with all the boxes drawn
    """
    print("\nObject detection on " + Path(image_path).name)
    img = cv2.imread(image_path)
    HEIGHT, WIDTH, _ = [length // 32 * 32 for length in img.shape]  # get image size
    yolo_model = YOLOv4(input_shape=(HEIGHT, WIDTH, 3), anchors=YOLOV4_ANCHORS, num_classes=80, training=False,
                        yolo_max_boxes=100, yolo_iou_threshold=0.5, yolo_score_threshold=0.5,
                        weights="yolov4_utils/yolov4.h5")
    yolo_model.summary()
    image = tf.convert_to_tensor(img)
    image = tf.image.resize(image, (HEIGHT, WIDTH))
    images = tf.expand_dims(image, axis=0) / 255.0
    boxes, scores, classes, valid_detections = yolo_model.predict(images)
    if os.path.isdir(output_path) is False:
        os.mkdir(output_path)
    output_image_path = os.path.join(output_path, Path(image_path).name)
    result_img = get_processed_image(img, boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT], scores[0], classes[0].astype(int))
    cv2.imwrite(output_image_path, result_img)
    return output_image_path


def detect_video(video_path, output_path="output_videos"):
    """
        Detects the objects from a video using a YOLOv4 model and returns the result video with the same name

        Arguments:
            video_path -- path of the image to run detection on
            output_path -- the path of the output folder

        Returns:
            output_video_path -- path to processed video with all the boxes drawn
    """
    print("\nObject detection on " + Path(video_path).name)
    cap = cv2.VideoCapture(video_path)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    WIDTH, HEIGHT = [length // 32 * 32 for length in frame_size]
    yolo_model = YOLOv4(input_shape=(HEIGHT, WIDTH, 3), anchors=YOLOV4_ANCHORS, num_classes=80,
                        training=False, yolo_max_boxes=100, yolo_iou_threshold=0.5, yolo_score_threshold=0.5,
                        weights="yolov4_utils/yolov4.h5")
    yolo_model.summary()
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    if os.path.isdir(output_path) is False:
        os.mkdir(output_path)
    output_video_path = os.path.join(output_path, Path(video_path).name)
    out = cv2.VideoWriter(output_video_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), frame_size)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            image = tf.convert_to_tensor(frame)
            image = tf.image.resize(image, (HEIGHT, WIDTH))
            images = tf.expand_dims(image, axis=0) / 255.0
            boxes, scores, classes, valid_detections = yolo_model.predict(images)
            frame = get_processed_image(frame, boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT], scores[0], classes[0].astype(int))
            cv2.imshow(Path(video_path).name, frame)
            out.write(frame)
            if cv2.waitKey(int(200 // cap.get(cv2.CAP_PROP_FPS))) & 0xFF == 27:  # 27 = ESC ASCII code
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_video_path


vid_path = "videos/cars_small.mp4"
os.startfile(os.path.normpath(vid_path))
output_vid_path = detect_video(vid_path)
os.startfile(output_vid_path)

img_path = get_file("cars.jpg", "https://github.com/sicara/tf2-yolov4/raw/master/notebooks/images/cars.jpg", cache_dir="images/", cache_subdir="")
os.startfile(os.path.normpath(img_path))
output_img_path = detect_image(img_path)
os.startfile(output_img_path)
