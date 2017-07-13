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
<img src="https://github.com/willofdiamond/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/test_class_spread.PNG" alt="Drawing" style="width: 200px;"/>
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

I used the Numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

I had used pandas library to extract signal names from CSV file

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I had attached a histogram of the spread of classes in a histogram for train, validation and test set.

![Train spread][image1] | ![Valid spread][image2] | ![Test spread][image3]

It can be observed that the distribution of the train, validation and test set is similar.
All the 43 Classes does not have a uniform distribution in the data set.

###Design and Test a Model Architecture

#### Steps in preprocessing

1. As a first step, I converted the images to grayscale because RGB channels are sensitive to lighting conditions. When compared the results of grey channel images with RGB channels, RGB channels tend to give much better results proving that color information also plays a vital role in classification. This made me choose RGB channel images as my input. Exploring other color spaces may result in better results.
2. I had normalized the images to get all features on the same scale.




#### Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5x6     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5x16   | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|							|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten		| input  5x5x16  | 400     									|
| Fully connected		| input  400  | 120    									|
| RELU					|							|
| dropout| |
| Fully connected		| input  120  | 84   									|
| RELU					|							|
| dropout| |
| Fully connected		| input  84  | 434  									|
| Softmax				|         									|




#### 3. Model Parameters
For training my model, I used 10 epochs, batch size of 300 ,learning rate of 0.005 and drop out of 0.8.



#### 4. Results
If an iterative approach was chosen:
* Initially I converted RGB images in to grey scale image and normalized it to zero mean and unit standard deviation image. But RGB images tend give better results for same tuned parameters. So I sticked with RGB channels as input.
* My architecture was based on LeNet.As LeNet was designed for digit classification and most of the Traffic signs have digits I believe this is the best architecture to start.
* I had adjusted Batch size, dropout and epoch parameters to get a validation accuracy of 0.94 along with preprocessing.



My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.94
* test set accuracy of 0.933




### Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web and resized to 32x32x3: This images after resizing mostly seems comparable to there original images but Slippery Road image seems very close to road work symbol. I believe this will be a hard classification while the other images should be easy to classify.

![Test image 1][image4] ![Test image 2][image5] ![Test image 3][image6]
![Test image 4][image7] ![Test image 5][image8] ![Test image 6][image9]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction with grey scale images.

| Image			        |     Prediction	        					| Top 5 class probability | Top 5 classes prediction|
|:---------------------:|:---------------------------------------------:|:---------------------------------------------:|:---------------------------------------------:|
| Speed limit (60km/h)      		| Speed limit (60km/h)  	| [  9.94940996e-01   4.70735831e-03   1.57816874e-04   8.65524999e-05 8.33368831e-05]	| [ 3  2 16  9  5]|
| ahead     			| ahead										|   [  1.00000000e+00   1.58982274e-16   1.59065669e-20   1.27800665e-20 1.06318447e-21]  | [35 33  9 34 37 ]   |
| right turn					| right turn			|		[ [  1.00000000e+00   1.50360003e-14   8.40688247e-16   4.85597361e-19 2.64132421e-20]	  |  [33 40 39 34 38]     |
| road work	      		| road work				|	 	[  1.00000000e+00   1.70044737e-30   0.00000000e+00   0.00000000e+00 0.00000000e+00]			|    [[25 29  0  1  2 ]  |
| 	Slippery Road	|    Slippery Road   		|			[  9.99842644e-01   1.40188931e-04   9.14557313e-06   5.83448582e-06 1.66681025e-06]		|  [23 29 31 25 19]  |      |
| Stop sign		| stop sign      			|		[  1.00000000e+00   2.45443050e-17   2.21973190e-22   1.35187295e-22 1.32074324e-22]		|     [14 15  3 17 13] |



My model was able to correctly guess all the six traffic signs, which gives an accuracy of 100%. This is impressive considering the images has less resolution and signs like road work and slippery road are similar.
