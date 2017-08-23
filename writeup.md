**Traffic Sign Recognition** 

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

***Data Set Summary & Exploration***

I used python and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing distribution of classes in the training, validation and test set.

![alt text][image1]

![alt text][image2]

![alt text][image3]

***Design and Test a Model Architecture***

As a first step, I decided to convert the images to grayscale because color is not an important feature here. Also it is possible to have different blue and red tints on sign.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]
![alt text][image5]

After that I normalized the image data with Tensorflow standardization function to have zero mean and unit norm.

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

To train the model, I used Adam optimizer with learning rate 0.001. Batch size is 128 and number of epochs is 50.

My final model results were:
* training set accuracy of 0.975
* validation set accuracy of 0.968 
* test set accuracy of 0.939

I started with LeNet model and my test set accuracy was lower then 0.93. I decided to add dropout layer after each fully connected layer because this helps with overfitting and I used AlexNet as example.
 

***Test a Model on New Images***

I decided to use eight images.

![alt text][image6]

Speed limit (30km/h), Speed limit (60km/h) and Speed limit (70km/h) images were selected because they are very close to each other (geometry and colors) and I want to check how model works with such images.

Double curve image was selected because there are two kinds of this sign - left and right and I want to check if model can predict it.

Other images were selected to check how model works in general.

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


The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%.  For each of 8 images model make prediction with very high probability. 

Speed limit (30km/h)

| Class | Probability 	|
|:-----:|:-------------:|
|  1	| 0.999968		|
|  2 	| 0.000032 		|
| 14 	| 0.000000 		|
|  0 	| 0.000000		|
|  5 	| 0.000000 		|

Turn right ahead

| Class | Probability 	|
|:-----:|:-------------:|
| 33 	|  1.000000		|
| 35 	|  0.000000		|
| 17 	|  0.000000		|
| 13 	|  0.000000		|
| 39 	|  0.000000		|

Stop

| Class | Probability 	|
|:-----:|:-------------:|
| 14	| 0.999558
|  4	| 0.000225
| 34	| 0.000077
| 17	| 0.000059
| 37	| 0.000025

Double curve

| Class | Probability 	|
|:-----:|:-------------:|
| 28	| 1.000000
| 20	| 0.000000
| 11	| 0.000000
| 23	| 0.000000
|  0	| 0.000000

Speed limit (60km/h)

| Class | Probability 	|
|:-----:|:-------------:|
|  3	| 1.000000
|  5	| 0.000000
|  2	| 0.000000
| 35	| 0.000000
| 14	| 0.000000

Speed limit (70km/h)

| Class | Probability 	|
|:-----:|:-------------:|
|  4	| 1.000000
|  1	| 0.000000
|  0	| 0.000000
| 14	| 0.000000
| 39	| 0.000000

Priority Road

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


*** (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)***

Here is how feature maps looks like when we use first image from test set.

![alt text][image7]

![alt text][image8]

First convolution layer extracts information about geometry details like round border. It is hard to say something about second convolution layer because feature maps looks abstract for me.
