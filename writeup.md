# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center.jpg "Center driving"
[image2]: ./images/flip.png "normal image"
[image3]: ./images/flipped.png "flipped image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is inspired by the NVIDIA convolutional network. This model starts with a normalization layer (keras lambda
 layer). Then there are 5
 convolutional layers with strides of both 3 and 5 and depths ranging between 24 and 64.
 Finally, it has 3 fully connected layers. 
  
  In a second moment, I added 2
  dropout layers after the first 2 fully connected layers to fight
  overfitting.

The model is defined in model.py from line 47 to line 61.

#### 2. Attempts to reduce overfitting in the model and train/valid/test split

The model contains dropout layers in order to reduce overfitting as described above.

The dataset is divided in training and test data (line 43 of model.py), the test data is the 10% of the full dataset. Moreover I took 20% percent of the training data and used it for the validation set (line63 of model.py). After the model is trained I performed a test (lines 67 and 68) that showed that the model had learned well from the data. This was confirmed by the fact that the vehicle was actually able to drive around the circuit.
#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 63).

#### 4. Appropriate training data

For details about how I created the training data, see the next section. However, I finally decided to use the given
 dataset for the training of the model. Indeed, I found it really hard to drive around the track without using a
  joystick. When using the already recorded dataset I set the number of epochs to 1, since the model started
   overfitting right away. Anyway, with this data (converted to RGB) the model was able to infer the proper angle to
    keep the car in the middle of the road.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple but functioning architecture and
 then increase the accuracy gradually.

My first step was to use a network with a convolutional layer and a fully connected layer. I thought this model might be
 a good starting point
 because we have to deal with images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To fight the overfitting, I modified the model so that it was depeer. I the added 2 convolutional layers and 2 other
 fully connected layers. Although I saw some improvement I also added two dropout layers, which improved the accuracy
  of the validation set. Additionally, I augmented the dataset by flipping the images.

Another step that decreased dramatically the error was to add a lambda layer to the network. This layer normalized
 the images and dropped the top and bottom part of it, keeping only the meaningful information contained in the images.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots
 where the vehicle fell off the track. Especially once the vehicle started to drift off , it wasn't able to recover
  to the center of the lane. To improve the driving
  behavior in
  these cases, I loaded also the left and right images contained in the dataset and applied manually a correction
   factor to the steering angle so that to the image to the right a negative correction was applied and a positive
    one on the left camera image.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

input: 160x120x3
1. Lambda layer
2. Cropping layer
3. Conv layer (stride: 5, subsample:2, depth: 24)
4. Conv layer (stride: 5, subsample:2, depth: 36)
5. Conv layer (stride: 5, subsample:2, depth: 48)
6. Conv layer (stride: 3, subsample:0, depth: 64)
7. Conv layer (stride: 3, subsample:0, depth: 64)
8. Dense layer (output: 100)
9. Dropout layer (prob: 0.5)
10. Dense layer (output: 50)
11. Dropout layer (prob: 0.25)
12. Dense layer (output: 10)
13. Dense layer (output: 1)

output: 1x1x1


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. Here is an
 example image of center lane driving:

![alt text][image1]

To augment the data set, I flipped images and angles thinking that this would help the model to generalize. For example
, here
 is an image
 that has then been flipped:

![alt text][image2]
![alt text][image3]

After the collection process, I had 10728 number of data points. I then preprocessed this data by normalizing and
 cropping the images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under
 fitting. The ideal number of epochs was 2 and I used an adam optimizer so that manually training the learning rate
  wasn't necessary.
  
Finally, even though the accuracy of the model was high both for the training and the validation set, the car wouldn
't stay on track. This is due to the fact that I really was a terrible driver! It was really difficult to drive
 around the circuit using the keyboard. I therefore decided to make use of the provided dataset only. With that, the
  car was able to drive around the track statying in the center of the lane most of the time.