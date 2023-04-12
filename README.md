# Real-time SRCNN implementation using darknet framework #
* Darknet is an open source neural network framework written in C and CUDA, https://github.com/pjreddie/darknet.
* The main purpose of Darknet is object detection, especially for a detection algorithm, YOLO.
* The goal of this project is to implement a real-time video enhencing deep learning architecture using Darknet framework with GPU, CUDA, CUDNN and OPENCV.
* SRCNN, Image Super-Resolution Using Deep Convolutional Networks (https://arxiv.org/abs/1501.00092) algorithm is implemented in this repository. 
* For faster inference implementatinon, multi-threadings are applied in multiple spot, e.g., data input loading, output image convertin. 

# I made some demo videos and those are on Youtube #
* Video resolution upscaled from 640x360 to 1920x 1080: https://www.youtube.com/watch?v=qEWNTEBMlk4
* Video resolution upscaled from 352x240 to 1056x720: https://www.youtube.com/watch?v=z3EA23KzQvM


## How to compile
* To compile, follow instrutions below.
* Install make, CUDA, OPENCV and CUDNN
* git clone https://github.com/hjun2020/image_enhencer_project.git and checkout to the srcnn_demo branch, git checkout srcnn_demo 
* Then, type make in your CLI

## How to run network for some sample videos
* You can try image enhencing for some sample videos:
* Type ./darknet enhancer srcnn_video_demo cfg/voc.data cfg/srcnn1.cfg backup/srcnn1.backup_test data/people_crossing_640x360.mp4 in the command line.


## How to run training
* You can train SRCNN neural net using VOC data
* To download and set up training data, follow the instruction "Training YOLO on VOC" in https://pjreddie.com/darknet/yolo/.
* Then, run ./darknet enhencer train cfg/voc.data cfg/srcnn1.cfg backup/srcnn1.backup_test in your command line interface.
* After each 100 epochs, the trained model will be saved at backup/ directory.
