# Python program to explain cv2.rectangle() method

# importing cv2
import cv2
from random import randint


def test_initial_boxes(frame, boxes, colors):
    pass

    for b in boxes:
        cv2.rectangle(frame,
                      (b[0], b[1]),
                      (b[0] + b[2], b[1] + b[3]),
                      colors[randint(0, len(colors)-1)])

    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
