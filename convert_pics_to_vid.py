import cv2
import os

image_folder = 'VisDrone2019-MOT-train/sequences/uav0000013_00000_v'
video_name = './converted_videos/uav0000013_00000_v'
extension = ".mp4"

curr_video_name = video_name + extension

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(curr_video_name, fourcc, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
