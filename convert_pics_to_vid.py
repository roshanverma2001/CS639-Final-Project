import cv2
import os



def convert_video(source_directory = "VisDrone2019-MOT-train/sequences/", target_directory = "./converted_videos_train/"):
   
    extension = ".mp4"

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')


    count = 0
    for filename in os.listdir(source_directory):
        image_folder = os.path.join(source_directory, filename)
        target_video_name = os.path.join(target_directory, filename)
        target_video_name = target_video_name + extension

        
        if os.path.exists(target_video_name):
            print(target_video_name, "exists")
            continue

       # print("Folder {} has target {}".format(image_folder, target_video_name))
        # print(image_folder)
      
        images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(target_video_name, fourcc, 1, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))
        print("Done with : ", target_video_name)
        count += 1

        cv2.destroyAllWindows()
        video.release()
    
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
