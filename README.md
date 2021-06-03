# Face-Mask-Detector

COVID-19: Face Mask Detector, developed  a detection Model with 99% accuracy in training & testing. Automatically detect whether a person is wearing a mask or not in real-time video streams.

![image](https://github.com/alinasahoo/Face-Mask-Detector/blob/main/examples/Alina_FM3.PNG)
![image](https://github.com/alinasahoo/Face-Mask-Detector/blob/main/examples/Alina_FM4.PNG)

![image](https://github.com/alinasahoo/Face-Mask-Detector/blob/main/examples/Alina_FM1.PNG)
![image](https://github.com/alinasahoo/Face-Mask-Detector/blob/main/examples/Alina_FM2.PNG)

Table of Content
----------------
* Overview
* Motivation
* Installation
* Features
* Project structure
* Result/Summary
* Future scope of project

Overview / What is it ??
------------------------
* COVID-19: Face Mask Detector with OpenCV, Keras/TensorFlow, and Deep Learning
* Automatically detects whether a person is wearing a mask or not in real-time video streams.
* Our goal is to train a custom deep learning model to detect whether a person is or isn't wearing a mask.

Motivation  ??
---------------
* Our goal is to train a custom deep learning model to detect whether a person is or is not wearing a mask.
* Rapid increase of covid leading to the importance of wearing a mask during the pandemic at these times also has been increased.
* Universal mask use can significantly reduce virus transmission in communities.
* Masks and face coverings can prevent the wearer from transmitting the COVID-19 virus to others and may provide some protection to the wearer. 

Installation / Tech Used
------------------------------
* Dataset : Real World Masked Face Dataset (RMFD)
* AI/DL Techniques/Libaries : OpenCV, Keras/TensorFlow, MobileNetV2

Features
---------
![image](https://user-images.githubusercontent.com/41515202/94375410-098a0e80-0131-11eb-8a9f-2a2df72359e7.png)

1. Dataset consists of 1,376 images belonging to two classes:
    with_mask : 690 images
    without_mask : 686 images
    
2. Facial landmarks allow us to automatically infer the location of facial structures, including - Eyes, Eyebrows, Nose, Mouth, Jawline.

3. Once we know where in the image the face is, we can extract the face Region of Interest (ROI).

   ![image](https://user-images.githubusercontent.com/41515202/94375362-bdd76500-0130-11eb-9fc3-73b67e2ad162.png)

4. And from there, we apply facial landmarks, allowing us to localize the eyes, nose, mouth, etc.

   ![image](https://user-images.githubusercontent.com/41515202/94375376-d8a9d980-0130-11eb-8a62-8341b74fe2a3.png)

5. Next, we need an image of a mask (with a transparent background) such as the one below:

   ![image](https://user-images.githubusercontent.com/41515202/94375387-ee1f0380-0130-11eb-8a2c-0417b5de4288.png)

6. This mask will be automatically applied to the face by using the facial landmarks (namely the points along the chin and nose) to compute where the mask will be placed.
The mask is then resized and rotated, placing it on the face:

   ![image](https://user-images.githubusercontent.com/41515202/94375396-f9722f00-0130-11eb-82c4-e5ef61d86e2f.png)

7. Final Dataset with & without mask will be:

   ![image](https://user-images.githubusercontent.com/41515202/94375422-16a6fd80-0131-11eb-992c-07f7caa4862e.png)

8. Two-phase COVID-19 face mask detector:

   ![image](https://user-images.githubusercontent.com/41515202/94375426-27f00a00-0131-11eb-82ac-11e28d0b0d95.png)

9. In order to train a custom face mask detector, we need to break our project into two distinct phases, each with its own respective sub-steps (as shown by Figure 1 above).

   *	Training: Here we’ll focus on loading our face mask detection dataset from disk, training a model (using Keras/TensorFlow) on this dataset, and then serializing the face mask detector to disk.
   *	Deployment: Once the face mask detector is trained, we can then move on to loading the mask detector, performing face detection, and then classifying each face as with_mask or without_mask

STEPS/REQUIREMENTS
-----------------------
1. Data extraction
2. Building the Dataset class
3. Building our face mask detector model
4. Training our model
5. Testing our model on real data -> IMAGE/VIDEO
6. Results

PROJECT STRUCTURE
-------------------------
![image](https://user-images.githubusercontent.com/41515202/94375435-35a58f80-0131-11eb-99a2-e9e74ccb911f.png)

Require 3 Python scripts:
-----------------------------
* train_mask_detector.py :  Accepts our input dataset and fine-tunes MobileNetV2 upon it to create our mask_detector.model. A training history plot.png. Containing accuracy/loss curves is also produced
* Detect_mask_video.py : Using your webcam, this script applies face mask detection to every frame in the stream.
* Detect_mask_image.py : Performs face mask detection in static images.

Next two sections, we will train our face mask detector.
-----------------------------------------------------------
1. Implementing our COVID-19 face mask detector training script with Keras and TensorFlow:
* we’ll be fine-tuning the MobileNet V2 architecture, a highly efficient architecture that can be applied to embedded devices with limited computational capacity (ex., Raspberry Pi, Google Coral, NVIDIA Jetson Nano, etc.)
* Reason : Deploying our face mask detector to embedded devices could reduce the cost of manufacturing such face mask detection systems, hence why we choose to use this architecture.
2. Training the COVID-19 face mask detector with Keras/TensorFlow
3. Implementing our COVID-19 face mask detector for images with OpenCV
4. COVID-19 face mask detection in images with OpenCV
5. Implementing our COVID-19 face mask detector in real-time video streams with OpenCV
6. Detecting COVID-19 face masks with OpenCV in real-time Video Streams 

STEPS FOR TRAINING MODEL
------------------------------
1. From there, open up a terminal, and execute the following command:
   $ python train_mask_detector.py --dataset dataset
 
![image](https://user-images.githubusercontent.com/41515202/97676743-16df4380-1ab7-11eb-9463-cfd5ad4e4b3a.png)

2. Training accuracy/loss curves demonstrate → high accuracy and little signs of overfitting on the data.
3. Obtained ~99% accuracy on our test set.
4. Looking at Figure , we can see there are little signs of overfitting, with the validation loss lower than the training loss.
5. Given these results, we are hopeful that our model will generalize well to images outside our training and testing set.

SUMMARY/RESULT
---------------
* Developed detection Model with 97% accuracy, automatically detect person is wearing a mask or not in real-time video streams.
* Extracted the face ROI & facial landmarks. And applied to the face by using the facial to compute.
* Used the highly efficient MobileNetV2 architecture & fine-tuned MobileNetV2 on our mask/no-mask dataset and obtained a classifier that is 97% accurate.
* Determined the class label encoding based on probabilities associated with color annotation.

Future Scope
------------
* Can be used in CCTV cameras for capturing wide peoples or group of peoples.
* Can be improved for further as per requiremnts.
