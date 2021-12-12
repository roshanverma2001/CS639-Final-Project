from label_videos import *

def main():
   label_all_videos(source_directory = "videos_train", destination_directory="videos_train_labeled", confidence= 0.2)


if __name__ == "__main__":
    main()