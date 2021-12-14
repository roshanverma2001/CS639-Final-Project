from label_videos import *

def main():
  figure_out_accuracy_all_videos(source_directory = "videos_test", destination_directory="videos_test_labeled", confidence= 0.5)
   #label_all_videos(source_directory = "videos_train", destination_directory="videos_train_labeled", confidence= 0.2)
   #label_all_videos(source_directory = "videos_test", destination_directory="videos_test_labeled", confidence= 0.5)
  # figure_out_accuracy_all_videos(source_directory = "videos_train", destination_directory = "videos_train_labeled", confidence = 0.5)



if __name__ == "__main__":
    main()
