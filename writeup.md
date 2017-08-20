#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization_train.png "Train Set Visualization"
[image2]: ./examples/visualization_validation.png "Validation Set Visualization"
[image3]: ./examples/visualization_test.png "Test Set Visualization"
[image4]: ./examples/original_image.png "Original Image"
[image5]: ./examples/grayscale_image.png "Grayscale Image"
[image6]: ./examples/traffic_signs.png "Traffic Signs"
[image7]: ./examples/conv1_features.png "Conv1 Layer Features"
[image8]: ./examples/conv2_features.png "Conv2 Layer Features"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ivan-ushakov/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing distribution of classes in the training, validation and test set.

![alt text][image1]
![alt text][image2]
![alt text][image3]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color is not an important feature here. Also it is possible to have different blue and red tints on sign.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]
![alt text][image5]

As a last step, I normalized the image data with Tensorflow standardization function to have zero mean and unit norm.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6		|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6	  				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x6		  				|
| Fully connected		| input 400, output 120							|
| RELU					|												|
| Dropout				|												|
| Fully connected		| input 120, output 94 							|
| RELU					|												|
| Dropout				|												|
| Fully connected		| input 84, output 43							|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam optimizer with learning rate 0.001. Batch size is 128 and number of epochs is 50.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.975
* validation set accuracy of 0.968 
* test set accuracy of 0.939

I started with LeNet model and my test set accuracy was lower then 0.93. I decided to add dropout layer after each fully connected layer because this helps with overfitting and I used AlexNet as example.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I decided to use eight images.

![alt text][image6]

I selected Speed limit (30km/h), Speed limit (60km/h) and Speed limit (70km/h) images because they are very close to each other and I want to check how model works with such images.

I selected Double curve image because there are two kinds of this sign - left and right and I want to check if model can predict it.

Other images were selected to check how model works in general.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        				|     Prediction	        					| 
|:-------------------------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)					| Speed limit (30km/h) 							| 
| Turn right ahead						| Turn right ahead 								|
| Stop									| Stop											|
| Double curve     						| Children crossing				 				|
| Speed limit (60km/h)					| Speed limit (60km/h)      					|
| Speed limit (70km/h)					| Speed limit (70km/h)      					|
| Priority Road							| Priority Road      							|
| End of all speed and passing limits	| End of all speed and passing limits  			|


The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For each of 8 images model make prediction with very high probability. 

Speed limit (30km/h) ts_1.jpg

| Class | Probability 	|
|:-----:|:-------------:|
|  1	| 0.999968		|
|  2 	| 0.000032 		|
| 14 	| 0.000000 		|
|  0 	| 0.000000		|
|  5 	| 0.000000 		|

Turn right ahead ts_2.jpg

| Class | Probability 	|
|:-----:|:-------------:|
| 33 	|  1.000000		|
| 35 	|  0.000000		|
| 17 	|  0.000000		|
| 13 	|  0.000000		|
| 39 	|  0.000000		|

Stop ts_3.jpg

| Class | Probability 	|
|:-----:|:-------------:|
| 14	| 0.999558
|  4	| 0.000225
| 34	| 0.000077
| 17	| 0.000059
| 37	| 0.000025

Double curve ts_4.jpg

| Class | Probability 	|
|:-----:|:-------------:|
| 28	| 1.000000
| 20	| 0.000000
| 11	| 0.000000
| 23	| 0.000000
|  0	| 0.000000

Speed limit (60km/h) ts_5.jpg

| Class | Probability 	|
|:-----:|:-------------:|
|  3	| 1.000000
|  5	| 0.000000
|  2	| 0.000000
| 35	| 0.000000
| 14	| 0.000000

Speed limit (70km/h) ts_6.jpg

| Class | Probability 	|
|:-----:|:-------------:|
|  4	| 1.000000
|  1	| 0.000000
|  0	| 0.000000
| 14	| 0.000000
| 39	| 0.000000

Priority Road ts_7.jpg

| Class | Probability 	|
|:-----:|:-------------:|
| 12	| 1.000000
| 40	| 0.000000
|  2	| 0.000000
|  9	| 0.000000
| 15	| 0.000000

End of all speed and passing limits ts_8.jpg

| Class | Probability 	|
|:-----:|:-------------:|
| 32	| 0.999950
| 41	| 0.000050
| 12	| 0.000000
|  6	| 0.000000
|  1	| 0.000000

Even for Double curve image where error was made. I guess that because all images have very low quality it is hard to distinguish small elements like Children crossing and Double curve have. Also all images (including my images from web) were made by scaling down big image to 32x32 and this also introduce artifacts. 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here is how feature maps looks like when we use first image from test set.

![alt text][image7]
![alt text][image8]

First convolution layer extracts information about geometry details like round border. It is hard to say something about second convolution layer because feature maps looks abstract for me.
