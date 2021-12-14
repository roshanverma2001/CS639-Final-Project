import pandas as pd
import os


def compute_accuracy_metrics_single_file(file):

    df = pd.read_csv(file,
                     #  sep=" ", #or delim_whitespace=True, #separator is whitespace
                     header=None,  # no header
                     #   usecols=[1, 2, 3], #parse only 3,4,6 columns
                     names=['percentage_frames', 'avg_area_intersection',
                            'avg_percentage_intersection'],  # set columns names
                     )

   # print(df)

    file_header = file.split(".")[0]
    #print(f"For video {file_header}")

    avg_boxes_detected = df["percentage_frames"].mean()
    max_boxes_detected = df["percentage_frames"].max()

    avg_area_detected = df["avg_area_intersection"].mean()
    max_area_detected = df["avg_area_intersection"].max()

    avg_perc_area_detected = df["avg_percentage_intersection"].mean()
    max_perc_area_detected = df["avg_percentage_intersection"].max()

    # print(f"Over all frames, we correctly predicted the number of boxes: {avg_boxes_detected} precent of the time, with                       a max of{max_boxes_detected}")

    return (avg_boxes_detected, avg_area_detected, avg_perc_area_detected)


def compute_metrics_all_files(curr_set):

    file_path = os.path.join("accuracy", curr_set)

    overall_metrics = []
    for file in os.listdir(file_path):
        curr_file = os.path.join(file_path, file)

        returned_metrics = compute_accuracy_metrics_single_file(curr_file)

        returned_metrics = (file, returned_metrics[0],returned_metrics[1], returned_metrics[2])
        overall_metrics.append(returned_metrics)

    print("Final Output: ")
    df = pd.DataFrame(overall_metrics, columns=["Filename",
                                                "Percentage of Boxes Detected",
                                                "Average Area Intersection",
                                                "Average Percentage area detected"])

    
    print(df)
    output_file_path = os.path.join("accuracy", str(curr_set) + "_" + "final_results.csv")
    df.to_csv(output_file_path, index = False)