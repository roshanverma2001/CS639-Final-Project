import numpy as np
import argparse
import time
import cv2
import os

## steps to downoading YOLO: https://cloudxlab.com/blog/setup-yolo-with-darknet/
## https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

argument_parser = argparse.ArgumentParser()

argument_parser.add_argument("-i", "--image", required=True)
argument_parser.add_argument("-y", "--yolo", required=True)
argument_parser.add_argument("-c", "--confidence", type=float, default=0.5)
argument_parser.add_argument("-t", "--threshold", type=float, default=0.3)

arguments = vars(argument_parser.parse_args())
# YOLO model is trained on CoCo dataset.

path_to_labels = os.path.sep.join([arguments["yolo"], "data", "coco.names"])
labels = open(path_to_labels).read().strip().split("\n")

colors = np.random.randint(0, 255, size=(len(labels), 3),
                           dtype="uint8")

# load yolo
path_to_yolo_weights = os.path.sep.join([arguments["yolo"], "yolov3.weights"])
path_to_yolo_config = os.path.sep.join([arguments["yolo"], "cfg", "yolov3.cfg"])

print("Loading YOLO!: ...")
# cv2.dnn == OpenCV's deep neural network class
yolo_nn = cv2.dnn.readNetFromDarknet(path_to_yolo_config, path_to_yolo_weights)

image = cv2.imread(arguments["image"])
(height, width) = image.shape[:2]

# get the output layer

output_layer = yolo_nn.getLayerNames()

# print("output layer: ", output_layer)
# for i in yolo_nn.getUnconnectedOutLayers():
#     print(i)
output_layer = [output_layer[i - 1] for i in yolo_nn.getUnconnectedOutLayers()]

# preprocess using opencv

blob = cv2.dnn.blobFromImage(
    image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

yolo_nn.setInput(blob)

start_time = time.time()

output_layers = yolo_nn.forward(output_layer)

end = time.time()

print(f"YOLO took {end - start_time} seconds")

bounding_boxes = []
confidence_boxes = []
classifications = []

for layer in output_layers:
    print(f"layer: {layer} has {len(layer)} detections")
    for detection in layer:

        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence >= arguments["confidence"]:
            curr_box = detection[0:4] * \
                       np.array([width, height, width, height])
            (x_center, y_center, box_width, box_height) = curr_box.astype("int")

            curr_x = int(x_center - box_width / 2)
            curr_y = int(y_center - box_height / 2)

            bounding_boxes.append(
                [curr_x, curr_y, int(box_width), int(box_height)])
            confidence_boxes.append(float(confidence))
            classifications.append(class_id)

# perform non maxima suppresion?
nms_indices = cv2.dnn.NMSBoxes(
    bounding_boxes, confidence_boxes, arguments["confidence"], arguments["threshold"])

print(f"len(nms_indices): {len(nms_indices)}")
if len(nms_indices) > 0:

    for idx in nms_indices.flatten():
        (box_x, box_y, box_width, box_height) = (bounding_boxes[idx][0], bounding_boxes[idx][1],
                                                 bounding_boxes[idx][2], bounding_boxes[idx][3])

        # draw box
        # print(f" {type(classifications)} || classifications: {classifications[1]}")
        # print(f"colors: {colors}")
        # print(f"idx: {idx}")
        curr_color = np.random.randint(0, 255, 3)
        curr_color = (int(curr_color[0]), int(curr_color[1]), int(curr_color[2]))

       # print(curr_color)
        cv2.rectangle(image,
                      (box_x, box_y),
                      (box_x + box_width, box_y + box_height),
                      curr_color,
                      2)

        print(f"idx: {idx} || classificaitons: {classifications} || labels: {labels}")
        text = f"{labels[classifications[idx]]}: {confidence_boxes[idx]}"

        cv2.putText(img = image, text = text, org = (box_x, box_y - 5),
                    fontFace= cv2.FONT_HERSHEY_PLAIN,
                    fontScale= 1,
                    color = curr_color, thickness= 2)

    cv2.imshow("Image: ", image)
    cv2.waitKey(0)
