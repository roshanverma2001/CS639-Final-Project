import sys
import cv2
from random import randint

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


def returnTracker(requested_tracker):
    if int(major_ver) < 4 and int(minor_ver) < 3:
        print("major < 4")
        tracker = cv2.cv2.Tracker_create(requested_tracker)

    try:
        opencv_trackers = {
            'BOOSTING':  cv2.legacy_TrackerBoosting.create(),
            'MIL': cv2.TrackerMIL_create(),
            'KCF': cv2.TrackerKCF_create(),
            #   'TLD': cv2.TrackerTLD_create(),
            'MEDIANFLOW': cv2.legacy_TrackerMedianFlow.create(),
            #   'GOTURN': cv2.Tracker_GOTURN_create(),
            'MOSSE':  cv2.TrackerCSRT_create(),
            'CSRT': cv2.legacy_TrackerMOSSE.create()
        }
        return opencv_trackers[requested_tracker.upper()]

    except KeyError:
        print("Sorry,you requested a tracker that I do not know about")
        print("Here is a list of available trackers: ", *opencv_trackers.keys())
