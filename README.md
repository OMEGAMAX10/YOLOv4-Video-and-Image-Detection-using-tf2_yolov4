# YOLOv4-Video-and-Image-Detection-using-tf2_yolov4
YOLOv4 Video and Image Detection using tf2_yolov4 and opencv modules.

Inspired by this Google Colab notebook: https://colab.research.google.com/github/sicara/tf2-yolov4/blob/master/notebooks/YoloV4_Dectection_Example.ipynb

Follow the steps from the notebook in order to obtain yolov4.h5:
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT" -O yolov4.weights
$ rm -rf /tmp/cookies.txt
$ convert-darknet-weights yolov4.weights -o yolov4.h5
