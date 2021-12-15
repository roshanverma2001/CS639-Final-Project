from label_videos import *
from accuracy_metrics import *
def main():
   # figure_out_accuracy_all_videos(source_directory = "videos_test", destination_directory="videos_test_labeled", confidence= 0.5)
   #label_all_videos(source_directory = "videos_train", destination_directory="videos_train_labeled", confidence= 0.2)
   #label_all_videos(source_directory = "videos_test", destination_directory="videos_test_labeled", confidence= 0.5)
    # figure_out_accuracy_all_videos(source_directory = "videos_train", destination_directory = "videos_train_labeled", confidence = 0.5)

  #  compute_metrics_all_files("test")

    label_all_images(source_directory="VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev/sequences",
    destination_directory= "images_test_labeled")

if __name__ == "__main__":
    main()
