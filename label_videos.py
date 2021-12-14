import cv2
import os
import yolo_video_main
from test_accuracy import *

def figure_out_accuracy_all_videos(source_directory, destination_directory, yolo="darknet", confidence=0.5, threshold=0.3):

    print(f"Testing accuracy  videos in: {source_directory}")

    num_vids = len(sorted(os.listdir(source_directory)))

    for idx, filename in enumerate(sorted(os.listdir(source_directory))):
 
        input_video_name = os.path.join(source_directory, filename)
        output_video_name = os.path.join(destination_directory, filename)

  
        
         
        # if os.path.exists(output_video_name):
        #     print(output_video_name, "exists")
        #     continue

        # target_video_name = os.path.join(target_directory, filename)
        # target_video_name = target_video_name + extension

        # print(input_video_name)
        # print(output_video_name)
        # print("-" * 15)
        print(f"!! Testing accuracy for {input_video_name}... Video {idx} of {num_vids}!!")

        test_accuracy_of_video(input=input_video_name,
                                             output=output_video_name, 
                                             yolo=yolo, 
                                             confidence=confidence, 
                                             threshold=threshold)

      
        print(f"!! Finished Accuracy of {input_video_name} !!")
        print("-" * 20)

    print("Finished labeling all videos!")


def label_all_videos(source_directory, destination_directory, yolo="darknet", confidence=0.5, threshold=0.3):

    print(f"Labeling videos in: {source_directory}")

    num_vids = len(sorted(os.listdir(source_directory)))
    for idx, filename in enumerate(sorted(os.listdir(source_directory))):
  
        input_video_name = os.path.join(source_directory, filename)
        output_video_name = os.path.join(destination_directory, filename)

         
        # if os.path.exists(output_video_name):
        #     print(output_video_name, "exists")
        #     continue

        # target_video_name = os.path.join(target_directory, filename)
        # target_video_name = target_video_name + extension

        # print(input_video_name)
        # print(output_video_name)
        # print("-" * 15)
        print(f"!! Labeling {input_video_name}... Video {idx} of {num_vids}!!")

        yolo_video_main.convert_single_video(input=input_video_name,
                                             output=output_video_name, 
                                             yolo=yolo, 
                                             confidence=confidence, 
                                             threshold=threshold)
        print(f"!! Finished Labeling {input_video_name} !!")
        print("-" * 20)

    print("Finished labeling all videos!")



    #    all_images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])

    #     if os.path.exists(target_video_name):
    #         print(target_video_name, "exists")
    #         continue

    #    # print("Folder {} has target {}".format(image_folder, target_video_name))
    #     # print(image_folder)

    #     images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
    #     frame = cv2.imread(os.path.join(image_folder, images[0]))
    #     height, width, layers = frame.shape

    #     video = cv2.VideoWriter(target_video_name, fourcc, 1, (width,height))

    #     for image in images:
    #         video.write(cv2.imread(os.path.join(image_folder, image)))
    #     print("Done with : ", target_video_name)
    #     count += 1

    #     cv2.destroyAllWindows()
    #     video.release()

    print("Finished writing all to videos")

# image_folder = 'VisDrone2019-MOT-train/sequences/uav0000013_00000_v'
# video_name = './converted_videos/uav0000013_00000_v'
# extension = ".mp4"

# curr_video_name = video_name + extension

# images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])

# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape

# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# video = cv2.VideoWriter(curr_video_name, fourcc, 1, (width,height))

# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))

# cv2.destroyAllWindows()
# video.release()
