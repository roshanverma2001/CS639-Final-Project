import os
import cv2
import numpy as np
import imutils as imutils
import math
import matplotlib.pyplot as plt

from shapely.geometry import Polygon


def test_accuracy_of_video(input, output, yolo, confidence=0.5, threshold=0.5):

    arguments = {}
    arguments["input"] = input
    arguments["output"] = output
    arguments["yolo"] = yolo
    arguments["confidence"] = confidence
    arguments["threshold"] = threshold

    video_name = arguments["input"].split(".")[0].split("/")
   # print("video name: ", video_name)

    curr_type = video_name[0].split("_")[1]

    extension = ".txt"
    accuracy_write_path = os.path.join(
        "accuracy", curr_type, video_name[1]) + extension

   # print(f"accuracy_write_path: {accuracy_write_path}")

    accuracy_write_file = open(accuracy_write_path, "w")

    # YOLO model is trained on CoCo dataset.
    path_to_labels = os.path.sep.join(
        [arguments["yolo"], "data", "coco.names"])
    labels = open(path_to_labels).read().strip().split("\n")

    # load yolo
    path_to_yolo_weights = os.path.sep.join(
        [arguments["yolo"], "yolov3.weights"])
    path_to_yolo_config = os.path.sep.join(
        [arguments["yolo"], "cfg", "yolov3.cfg"])

    print("\tLoading YOLO!: ...")
    # cv2.dnn == OpenCV's deep neural network class
    yolo_nn = cv2.dnn.readNetFromDarknet(
        path_to_yolo_config, path_to_yolo_weights)

    # get the output layer

    output_layer = yolo_nn.getLayerNames()

    output_layer = [output_layer[i - 1]
                    for i in yolo_nn.getUnconnectedOutLayers()]

    # preprocess using opencv

    video_stream = cv2.VideoCapture(arguments["input"])

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
        print("\t[INFO] could not determine # of frames in video")
        print("\t[INFO] no approx. completion time can be provided")
        total = -1

    count = 0
    while True:
        if count % 50 == 0:
            print(f"\ton frame {count} of {total}")
        # read the next frame from the file
        (grabbed, frame) = video_stream.read()
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break
        # if the frame dimensions are empty, grab them
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
            #  print(f"layer: {layer} has {len(layer)} detections")
            for detection in layer:

                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                # print(class_id)
                confidence = scores[class_id]

                if confidence >= arguments["confidence"]:
                    curr_box = detection[0:4] * \
                        np.array([width, height, width, height])
                    (x_center, y_center, box_width,
                     box_height) = curr_box.astype("int")

                    curr_x = int(x_center - box_width / 2)
                    curr_y = int(y_center - box_height / 2)

                    bounding_boxes.append(
                        [curr_x, curr_y, int(box_width), int(box_height)])
                    confidence_boxes.append(float(confidence))
                    classifications.append(class_id)

        # perform non maxima suppresion?
        nms_indices = cv2.dnn.NMSBoxes(
            bounding_boxes, confidence_boxes, arguments["confidence"], arguments["threshold"])

        # print("boudning boxes: ", bounding_boxes)
        # print(f"len(nms_indices): {len(nms_indices)}")
        accuracy_measurements = accuracy(count, bounding_boxes=bounding_boxes, nms_indices=nms_indices,
                                         video_name=video_name)

        # (number_detected, average_percent_diff, average_less_than_15) = accuracy(count, bounding_boxes=bounding_boxes, nms_indices=nms_indices,
        # video_name=video_name)
        accuracy_measurements = list(accuracy_measurements)
        accuracy_measurements = ",".join(accuracy_measurements)
        accuracy_measurements += "\n"
        accuracy_write_file.write(accuracy_measurements)
        count += 1


def accuracy(frame, bounding_boxes, nms_indices,  video_name):
    all_boxes = [(bounding_boxes[idx][0] + 1, bounding_boxes[idx][1] + 1,
                  bounding_boxes[idx][2] + 1, bounding_boxes[idx][3] + 1) for idx in nms_indices.flatten()]

    # print(len(all_boxes))
    # print("video_name: ", video_name)

    annotations_file_name = ""
    extension = ".txt"
    if video_name[0].split("_")[1] == "train":
        annotations_file_name = os.path.join(
            "VisDrone2019-MOT-train", "annotations", video_name[1]) + extension
    else:
        annotations_file_name = os.path.join(
            "VisDrone2019-MOT-test-dev", "VisDrone2019-MOT-test-dev", "annotations", video_name[1]) + extension

    # print(annotations_file_name)
    all_annotations = []
    frame += 1
    with open(annotations_file_name) as file:
        for line in file:
           # print(line)
           # stripped = [int(i) for i in line.rstrip()]
            stripped = line.strip()
            # Offset by 1   \to avoid 0 errors
            stripped = [int(i) + 1 for i in stripped.split(",")]
           # stripped = [int(i) for i in line.split(line.strip(),)
            # print(stripped)
            if stripped[0] == frame:
                all_annotations.append(stripped[2:6])

    if len(all_annotations) == 0:
        return ("0", "0", "0")

    # print(all_boxes)
    # print("!" * 20)
    # print(all_annotations)
    # print(len(all_annotations))

    all_boxes = sorted(all_boxes, key=lambda x: (x[0], x[1]))
    all_annotations = sorted(all_annotations, key=lambda x: x[0])
    accuracy_less_than_15 = [None for i in range(
        min(len(all_boxes), len(all_annotations)))]
    accuracy_percentages = [None for i in range(
        min(len(all_boxes), len(all_annotations)))]

    intersections = []
    visited_annotations = []
    percent_area = []
    for i in all_boxes:
        best_box = (None, None)

       # print(i)
        # print([(i[0], i[1]),
        #               (i[0] + i[2], i[1]),
        #               (i[0], i[1] - i[3]),
        #               (i[0] + i[2], i[1] - i[3])
        #               ])
        # top left, top right, bottom left, bottom right
        expansion = 150
        i = [x + expansion for x in i]
        bb = Polygon([(i[0], i[1])  ,
                      (i[0] + i[2], i[1]),
                      (i[0], i[1] - i[3]),
                      (i[0] + i[2], i[1] - i[3])
                      ])
       # print(f"bb: {bb}")
        bb = bb.buffer(0)
        # print(f"bb.area: {bb.area}")
        # plt.plot(*bb.exterior.xy)

        for j in all_annotations:
            j = [x + expansion for x in j]
            if j in visited_annotations:
                continue
            ab = Polygon(
                [    (j[0], j[1]),
                      (j[0] + j[2], j[1]),
                      (j[0], j[1] - j[3]),
                      (j[0] + j[2], j[1] - j[3])])
         #   print(f"ab: {ab}")
            ab = ab.buffer(0)


            # polygon = Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])
            # print(f"polygon: {polygon}")
            # other_polygon = Polygon([(1, 1), (4, 1), (4, 3.5), (1, 3.5)])
            # intersection = polygon.intersection(other_polygon)
            # print(intersection.area)
            intersection = bb.intersection(ab) 
            intersection = intersection.area
           # print(f"intersection: {intersection}")
            if None in best_box or intersection > best_box[0]:
                best_box = (intersection, j)
            
        intersections.append(best_box[0])
        visited_annotations.append(best_box[1])
        #print(intersection / bb.area)
        percent_area.append(best_box[0] / bb.area)

    avg_intersection = sum(intersections) / len(intersections)
    avg_area_covered = sum(percent_area) / len(percent_area)

    # print(f"frame: {frame}")
    # print(intersections)
    # print(avg_intersection)
    return (str(len(all_boxes) / len(all_annotations)),
            str(avg_intersection),
           str(avg_area_covered))


    # for i in range(min(len(all_boxes), len(all_annotations))):
    #     # print(all_boxes[i])
    #     # print(all_annotations[i])
    #     bb_area = all_boxes[i][2] * all_boxes[i][3]
    #     annotated_area = all_annotations[i][2] * all_annotations[i][3]

    #     # print("bb area: ", bb_area)
    #     # print("annotated area: ", annotated_area)
    #     # print("-" * 20)
    #     area_diff_float = abs(annotated_area - bb_area) / annotated_area
    #     accuracy_percentages[i] = (area_diff_float)

    #     if area_diff_float <= .15:
    #         accuracy_less_than_15[i] = 1
    #     else:
    #         accuracy_less_than_15[i] = 0

    # total number objects detected, average difference, # annotations with differences < 15
  