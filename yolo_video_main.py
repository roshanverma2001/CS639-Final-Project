import imutils as imutils
import numpy as np
import argparse
import time
import cv2
import os
import math
import test_accuracy
## steps to downoading YOLO: https://cloudxlab.com/blog/setup-yolo-with-darknet/
## https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

def convert_single_video(input, output, yolo, confidence = 0.5, threshold = 0.5):

    arguments = {}
    arguments["input"] = input
    arguments["output"] = output
    arguments["yolo"] = yolo
    arguments["confidence"] = confidence
    arguments["threshold"] = threshold

    video_name = arguments["input"].split(".")[0].split("/")
    print("video name: ", video_name)
    # YOLO model is trained on CoCo dataset.
    path_to_labels = os.path.sep.join([arguments["yolo"], "data", "coco.names"])
    labels = open(path_to_labels).read().strip().split("\n")

    colors = np.random.randint(0, 255, size=(len(labels), 3),
                            dtype="uint8")

    # print(colors)
    # print(len(colors))

    # load yolo
    path_to_yolo_weights = os.path.sep.join([arguments["yolo"], "yolov3.weights"])
    path_to_yolo_config = os.path.sep.join([arguments["yolo"], "cfg", "yolov3.cfg"])

    print("\tLoading YOLO!: ...")
    # cv2.dnn == OpenCV's deep neural network class
    yolo_nn = cv2.dnn.readNetFromDarknet(path_to_yolo_config, path_to_yolo_weights)

    # get the output layer

    output_layer = yolo_nn.getLayerNames()

    output_layer = [output_layer[i - 1] for i in yolo_nn.getUnconnectedOutLayers()]

    # preprocess using opencv

    video_stream = cv2.VideoCapture(arguments["input"])
    video_writer = None

    (height, width) = (None, None)

    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(video_stream.get(prop))
        print("\t[INFO] {} total frames in video".format(total))
    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        print("\tcould not determine # of frames in video")
        total = -1

    count = 0
    while True:
        if count % 50 == 0: print(f"\ton frame {count} of {total}")
        # read the next frame from the file
        (success, frame) = video_stream.read()
   
        if not success:
            break

        if width is None or height is None:
            (height, width) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                    swapRB=True, crop=False)
        yolo_nn.setInput(blob)
        output_layers = yolo_nn.forward(output_layer)

        bounding_boxes = []
        confidence_boxes = []
        classifications = []

        for layer in output_layers:
            for detection in layer:

 
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
        # helps with overlapping boxes
        nms_indices = cv2.dnn.NMSBoxes(
            bounding_boxes, confidence_boxes, arguments["confidence"], arguments["threshold"])

       
        if len(nms_indices) > 0:

            for idx in nms_indices.flatten():
                (box_x, box_y, box_width, box_height) = (bounding_boxes[idx][0], bounding_boxes[idx][1],
                                                        bounding_boxes[idx][2], bounding_boxes[idx][3])

                curr_color = tuple([int(c) for c in colors[class_id]])
                curr_color = (int(curr_color[0]), int(curr_color[1]), int(curr_color[2]))

                cv2.rectangle(frame,
                            (box_x, box_y),
                            (box_x + box_width, box_y + box_height),
                            curr_color,
                            2)

                text = f"{labels[classifications[idx]]}: {confidence_boxes[idx]}"

                cv2.putText(img=frame, text=text, org=(box_x, box_y - 5),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1,
                            color=curr_color, thickness=2)



            # check if the video writer is None
            if video_writer is None:
                # create our video writer
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                video_writer = cv2.VideoWriter(arguments["output"], fourcc, math.floor(total / 20),
                    (frame.shape[1], frame.shape[0]), True)
            count +=1
        # write the output frame to disk
        video_writer.write(frame)
    print("\t cleaning up...")
    video_writer.release()
    video_stream.release()
