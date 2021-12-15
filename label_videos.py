import cv2
import os
import yolo_video_main
import yolo_main
from test_accuracy import *


def figure_out_accuracy_all_videos(source_directory, destination_directory, yolo="darknet", confidence=0.5, threshold=0.3):

    print(f"Testing accuracy  videos in: {source_directory}")

    num_vids = len(sorted(os.listdir(source_directory)))

    for idx, filename in enumerate(sorted(os.listdir(source_directory))):
        if idx < 8:
            continue
        input_video_name = os.path.join(source_directory, filename)
        output_video_name = os.path.join(destination_directory, filename)


        print(
            f"!! Testing accuracy for {input_video_name}... Video {idx + 1} of {num_vids}!!")

        test_accuracy_of_video(input=input_video_name,
                               output=output_video_name,
                               yolo=yolo,
                               confidence=confidence,
                               threshold=threshold)

        print(f"!! Finished Accuracy of {input_video_name} !!")
        print("-" * 20)

    print("Finished labeling all videos!")


def figure_out_accuracy_all_images(source_directory, destination_directory, yolo="darknet", confidence=0.5, threshold=0.3):

    print(f"Testing accuracy images in: {source_directory}")

    num_vids = len(sorted(os.listdir(source_directory)))
    for idx, filename in enumerate(sorted(os.listdir(source_directory))):
 
        
        input_image = os.path.join(source_directory, filename)
        output_video_name = os.path.join(destination_directory, filename)

        # target_video_name = os.path.join(target_directory, filename)
        # target_video_name = target_video_name + extension

        # print(input_video_name)
        # print(output_video_name)
        # print("-" * 15)
        print(
            f"!! Testing accuracy for {input_image}... image {idx + 1} of {num_vids}!!")

        test_accuracy_of_image(input=input_image,
                               output=output_video_name,
                               yolo=yolo,
                               confidence=confidence,
                               threshold=threshold)

        print(f"!! Finished Accuracy of {input_image} !!")
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


def label_all_images(source_directory, destination_directory, yolo="darknet"):


    for idx, sub_directory in enumerate(sorted(os.listdir(source_directory))):
        num_images = len(
            sorted(os.listdir(os.path.join(source_directory, sub_directory))))

        input_path = os.path.join(source_directory, sub_directory)
        output_path = os.path.join(destination_directory, sub_directory)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        elif len(sorted(os.listdir(output_path))) == len(sorted(os.listdir(input_path))):
            print("Skipping: ", input_path)
            continue

        print(f"Labeling videos in: {sub_directory}... image set {idx + 1} of {len(os.listdir(source_directory))}")
        print()
        for img_idx, image in enumerate(sorted(os.listdir(os.path.join(source_directory, sub_directory)))):

            input_image = os.path.join(source_directory, sub_directory, image)
            output_image = os.path.join(
                destination_directory, sub_directory, image)

            # if os.path.exists(output_video_name):
            #     print(output_video_name, "exists")
            #     continue

            # target_video_name = os.path.join(target_directory, filename)
            # target_video_name = target_video_name + extension

            # print(input_video_name)
            # print(output_video_name)
            # print("-" * 15)
            if img_idx % 50 == 0:
                print(
                    f"\t!! Labeling {input_image}... image {img_idx + 1} of {num_images}!!")

            yolo_main.label_image(input=input_image,
                                  output=output_image,
                                  yolo=yolo)

        print(f"!! Finished Labeling {sub_directory} !!")
        print("-" * 20)

    print("Finished labeling all videos!")

