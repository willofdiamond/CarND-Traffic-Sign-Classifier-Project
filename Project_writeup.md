#**Traffic Sign Recognition** 

## Writeup Template

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/train_class_spread.PNG "train spread"
[image2]: ./examples/valid_class_spread.PNG "valid spread"
[image3]: ./examples/test_class_spread.PNG "test spread"
[image4]: ./examples/new_train.PNG "Traffic Sign 1"
[image5]: ./examples/new_train2.PNG "Traffic Sign 2"
[image6]: ./examples/new_train3.PNG "Traffic Sign 3"
[image7]: ./examples/new_train4.PNG "Traffic Sign 4"
[image8]: ./examples/new_train5.PNG "Traffic Sign 5"
[image9]: ./examples/new_train6.PNG "Traffic Sign 6"
[image10]: ./examples/train_example_rgb.PNG "color image"
[image11]: ./examples/train_example_normalized.PNG "grey image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/willofdiamond/CarND-Traffic-Sign-Classifier-Project.git)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy  library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

I had used pandas library to extract signal names from CSV file

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I had attached a histogram of the spread of classes in a histogram for train, valisation and test set.

![Train spread][image1] | ![Valid spread][image2] | ![Test spread][image3]

It can be observed that the distribution of the train, validation and test set is similar. 
All the 43 Classes does not have a uniform distribution in the data set.
 
###Design and Test a Model Architecture

#### Steps in preprocessing

1. As a first step, I decided to convert the images to grayscale because RGB channesls are sensitive to light changes. Converting to other color spaces can be effective as well.
2. I had normalized the images to get all features on the same scale



Here is an example of a traffic sign image before and after grayscaling.

![RGB Test Image][image10] | ![Normalized grey Test Image][image11]


#### Model Architecture 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grey image   							| 
| Convolution 5x5x6     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												| 
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5x16   | 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|							|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten		| input  5x5x16  | 400     									|
| Fully connected		| input  400  | 120    									|
| RELU					|							|
| Deopout| |
| Fully connected		| input  120  | 84   									|
| RELU					|							|
| Deopout| |
| Fully connected		| input  84  | 434  									|
| Softmax				|         									|

 


#### 3. Model Parameters
For training my model, I used 10 epochs, batch size of 250 and drop out of 0.8.



#### 4. Results

My final model results were:
* training set accuracy of 0.989
* validation set accuracy of 0.933
* test set accuracy of 0.923

If an iterative approach was chosen:
* I had initially tried to feed RGB image and the with Lenet architecture and the results are terrible. 
* Later I had prepossessed the RGB images to a greay scale image
* I initially started with Lenet model and then adusted the parameter till satisfied results are obtained
* I had changed convolution layer windows, dropout and max pooling parameters


### Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web and resized to 32x32x3:

![Test image 1][image4] ![Test image 2][image5] !Test image 3][image6] 
![Test image 4][image7] ![Test image 5][image8] ![Test image 6][image9]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)      		| Speed limit (30km/h)  									| 
| ahead     			| ahead										|
| right turn					| right turn											|
| road work	      		| road work					 				|
| 	Slippery Road	|    road work   							|
| Stop sign		| stop sign      							|


The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66%. This varies with the test set accuracy of 92. But  probablity score of  predicted Speed limit (60km/h)  and  Speed limit (30km/h) are very close.
