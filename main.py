import create_tracker
import locate_objects
import cv2


def main():
    #  convert_video()
    tracker = create_tracker.returnTrackerName("Boosting")
    multi_tracker = cv2.legacy.MultiTracker_create()

    locate_objects.define_bounding_boxes(
        video_path="./converted_videos/uav0000013_00000_v.mp4",
        # annotations_path="VisDrone2019-MOT-train/annotations/uav0000013_00000_v.txt",
        annotations_path="uav0000013_00000_v.txt",
        tracker=tracker,
        multiTracker=multi_tracker)

    # locate_objects.initialize_multiTracker(tracker=tracker,multiTracker=multi_tracker,boundingBoxes=bounding_boxes, frame = frame)


if __name__ == "__main__":
    main()
