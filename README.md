# CS639-Final-Project


Urban Analysis using Computer Vision

Srinath Srinivasan, Nikhil Yarram, Roshan Verma

CS 639, Fall 2021

Crowd counting is a technique used to estimate the number of people in a particular scene. (image or video) This technique plays a crucial role in urban planning, public safety, intelligent transportation and even helps optimize services in the commercial sector. Furthemore, crowd counting can be used to estimate the population density of an area in real-time. The population density can be used in a plethora of urban analysis such as monitoring usage of streets, malls, parks, etc. Such data could be especially useful in analyzing the spread of COVID-19, enforcing regulations such as social distancing, and allotting necessary resources to areas that require it the most.  In this research project, we will use computer vision to train a machine learning model with video footage from the VisDrone dataset. Our goal is to be able to perform object identification, crowd counting and segmentation on a given scene environment.  

With the rapidly increasing population in urban cities, it's becoming progressively difficult to quantify the effects of urban development plans. Madison, for example, was one of the fastest growing cities in Wisconsin, with a 16% population increase over the past decade. Computer vision enables an automated process for the evaluation of expansion schemes. Moreover, it allows for large scale estimation of the viability and functionality proposed changes. With a dataset of over 200,000 frames, we will be able to analyze and provide solutions for common dysfunctionalities in an urban society, including but not limited to - pedestrian/bicyclist integration, environmental expansion, municipal infrastructure improvements, residential safety, and homelessness elimination. Urban cities around the globe face similar problems, and by implementing machine learning techniques, we'll attempt to train our algorithm to identify these issues in environments not limited by our dataset.

By combining object detection with frame segmentation, we can classify areas of interest both on the street level, and an urban structure level, empowering us to combine this knowledge to possibly automate solution providing, such as the development of certain walkways, improving landmark accessibility, and enabling new public transport opportunities. In essence, we expect this computer vision algorithm to better the lives of all urban city dwellers, though the possibilities this provides in a rural community are vast as well. As cities begin to run out of space, our tool can help rural communities plan their expansion better to accommodate people as they look for more space. 

To solve this problem, our group will be using the VisDrone Dataset, available here: https://github.com/VisDrone/VisDrone-Dataset. The dataset contains
288 video clips, formed by 261,908 frames, and 10,209 static images, all captured by drones
The data was taken from 14 cities, both urban and country in China, and includes objects and density

This data serves as the perfect source to solve the issue of Urban analysis, as it gives an accurate depiction of cities and the life inside of it. We will be able to analyze usage patterns of things like streets, and give an understanding of how many people pass through a certain area and in what means of transportation (walking or biking). We will also do image segmentation, which will give a visual understanding of how an environment is broken down. By doing this over a time interval, we will be able to get a temporal Urban analysis, which can be important for crafting city policies and in the construction of new ones. 

This project will contain several technical steps, including
Understanding the raw data and it’s form
Edge Detection to help detect different objects in the scene
Analysis of the individual scenes/frames to try and identify the number of objects in a certain class
Mapping those objects to a certain color to create a visual representation of the image segmentation










Rough Time Table

10/11/21
Project Proposal

10/26/21
Familiarization with Data and Algorithms needed to do object classification

11/9/21
Object Classification

11/23/21
Able to count / detect number of objects and locations in an image -- including crowd counting

12/1/21
Segmentation and design functionality tests

12/7/21
Final Project Done, including web page

## Multi Object Tracking walkthrough:

Here you will find many files and directories:

- accuracy
    - Lists the results for each frame of every video with the number of bounding boxes in that specific frame and area covered
- converted_videos_test:
    - The test images given converted to video
- darknet
    - an implementaiton of YOLO built on the darknet architecture
    - Please refer to this for information about how to download: https://pjreddie.com/darknet/yolo/
- videos_test_labeled
    - the videos containing our predicted labels
    - also on google drive

- Utility functions
    - accuracy_metrics - calculate the accuracy of all videos
- label videos:
    - the majority of the code that runs through the video and labels each frame
- main.py
    - can call the functions from there
- test accuracy
    - calculates the accuracy of a given frame
- yolo_video_main.py
    - calls YOLO on a video to help with labeling



# Steps 
- download visdrone data from github
- extract
- download Yolov3 trained on COCO (or train on your own data)
- call main.py

