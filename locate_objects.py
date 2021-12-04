import cv2
from random import randint
from test_initial_boxes import test_initial_boxes
import create_tracker


def define_bounding_boxes(video_path, annotations_path, tracker, multiTracker):
    captured_video = cv2.VideoCapture(video_path)

    suc, frame = captured_video.read()
    if not suc:
        print("was unable to read video")

    located_objects = {}
    colors = []
    for line in open(annotations_path, "r"):
        split_line = line.split(",")
        split_line = [int(i) for i in split_line]
        if split_line[1] not in located_objects:
            located_objects[split_line[1]] = split_line[2:6]
            colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))

#     print(located_objects)
#     test_initial_boxes(frame=frame,
#                        boxes=located_objects.values(),
#                        colors=colors)
#
# #
    boxes = located_objects.values()

    # Initialize tracker
    for box in boxes:
        # print(box)
        # cv2.rectangle(frame,
        #               (box[0], box[1]),
        #               (box[0] + box[2], box[1] + box[3]),
        #               colors[randint(0, len(colors) - 1)])
        multiTracker.add(create_tracker.returnTrackerName("CSRT"), frame, box)
    # cv2.imshow('image', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # Process video and track objects
    while captured_video.isOpened():
        success, frame = captured_video.read()
        if not success:
            break

        # get updated location of objects in subsequent frames
        success, boxes = multiTracker.update(frame)

        # draw tracked objects
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2,  colors[i])

        # show frame
        cv2.imshow('MultiTracker', frame)

        # quit on ESC button
       # key = cv2.waitKey(3000)  # pauses for 3 seconds before fetching next image

        if cv2.waitKey(500) & 0xFF == 27:  # Esc pressed
            break
            # return located_objects.values(), frame



#------------------------------------------
#

# # def initialize_multiTracker(tracker, multiTracker, boundingBoxes, frame):
# #     for box in boundingBoxes:
# #         multiTracker.add(tracker, frame, box)
#
#     # captured_video = cv2.VideoCapture(video_path)
#
#     # suc, frame = captured_video.read()
#     # if suc:
#     #     print("was unable to read video")
#
#     # ## Select boxes
#     # bboxes = []
#     # colors = []
#
#     # # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
#     # # So we will call this function in a loop till we are done selecting all objects
#     # while True:
#     # # draw bounding boxes over objects
#     # # selectROI's default behaviour is to draw box starting from the center
#     # # when fromCenter is set to false, you can draw box starting from top left corner
#     #     bbox = cv2.selectROI('MultiTracker', frame)
#     #     bboxes.append(bbox)
#     #     colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
#     #     print("Press q to quit selecting boxes and start tracking")
#     #     print("Press any other key to select next object")
#     #     k = cv2.waitKey(0) & 0xFF
#     #     if (k == 113):  # q is pressed
#     #         break
#
#     # print('Selected bounding boxes {}'.format(bboxes))
