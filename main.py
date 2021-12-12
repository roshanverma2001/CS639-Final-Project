import create_tracker
import locate_objects
import cv2
from convert_pics_to_vid import *
import zipfile


def main():
    # path_to_zip_file = "VisDrone2019-MOT-test-dev.zip"
    # directory_to_extract_to = "./VisDrone2019-MOT-test-dev"
    # with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    #     zip_ref.extractall(directory_to_extract_to)
    test_vids_src = "./VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev/sequences"
    test_vids_dest = "./converted_videos_test/"
    convert_video(test_vids_src, test_vids_dest)
    # tracker = create_tracker.returnTrackerName("Boosting")
    # multi_tracker = cv2.legacy.MultiTracker_create()

    # locate_objects.define_bounding_boxes(
    #     video_path="./converted_videos/uav0000281_00460_v.mp4",
    #     annotations_path="VisDrone2019-MOT-train/annotations/uav0000013_00000_v.txt",
    #   #  annotations_path="uav0000013_00000_v.txt",
    #     tracker=tracker,
    #     multiTracker=multi_tracker)

   # locate_objects.initialize_multiTracker(tracker=tracker,multiTracker=multi_tracker,boundingBoxes=bounding_boxes, frame = frame)


if __name__ == "__main__":
    main()
