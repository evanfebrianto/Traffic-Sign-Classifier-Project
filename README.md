# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[sample-data]: examples/sample-data.png "Sample Data"
[undistorted]: examples/undistorted.png "Undistorted"
[distorted]: examples/distorted.png "Distorted"
[test-images]: examples/test-images.png "Sample Data From Web"
[ahead-only]: examples/ahead-only.png "Ahead Only"
[conv1]: examples/conv1_activation.png "conv1_activation"
[conv2]: examples/conv2_activation.png "conv2_activation"
[image1]: examples/dataset-distribution.png "Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/evanfebrianto/Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used numpy and python to calculate summary statistics of the traffic signs dataset and seaborn to visualize:
* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed over 43 classes.

![alt text][image1]

Below is the example of how the dataset looks like.
![alt text][sample-data]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize the image to improve the accuracy and it gives me a positive result. Because some of the data are limited compared to the others, I decided to generate additional data. This can improve make the model more robust because it has more data to learn. 

To add more data to the the data set, I applied some small distortion effect to the dataset. 

Here is an example of an original image and an augmented image:

Original image:

![alt text][undistorted]

Augmented image: 

![alt text][distorted]

After I generated this augmented data, the dataset becomes much bigger.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32    				|
| Droprout				| keep_prob = 0.5   							|
| Fully connected		| Input 1600, outputs 120                       |
| RELU					|												|
| Droprout				| keep_prob = 0.5   							|
| Fully connected		| Input 120, outputs 84                         |
| RELU					|												|
| Droprout				| keep_prob = 0.5   							|
| Fully connected		| Input 84, outputs 43                          |
|                       |                                               |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer, a batch size of 128, 20 epochs, a learn rate of 0.001. Lastly, I used 0.5 of dropout rate for training to achieve highest validation accuracy.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.983 
* test set accuracy of 0.972

I started with original LeNet architecture as it has been provided from the previous lecture. I managed to achieve around 0.893 validation accuracy. However, it is not sufficient to meet the requirement from rubric point. I added the depth and also more layer to extract more features. I know this may result in overfitting, that is why I put dropout operations between the fully connected layers. Surprisingly, the result was very good on the validation and test data.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][test-images]

The difficulty from these images is that they have different size. I programmed to resize them to be 32x32 pixels so that it will fit to the model. On top of that, the third image is a bit darker compared to the others.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                            |     Prediction                            | 
|:-----------------------------------------:|:-----------------------------------------:| 
| Speed limit (30km/h)                      | Speed limit (30km/h)                      | 
| Right-of-way at the next intersection     | Right-of-way at the next intersection     |
| Priority road			        		    | Priority road 	  					    |
| Vehicles over 3.5 metric tons prohibited	| Vehicles over 3.5 metric tons prohibited  |
| General caution			                | General caution                       	|
| Turn right ahead			                | Turn right ahead                       	|
| Keep right    			                | Keep right                             	|
| Keep right    			                | Keep right                               	|


The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 97.2%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Below is the probablity after I used softmax.

The first image
* Speed limit (30km/h): 1.00000
* Speed limit (20km/h): 0.00000
* Speed limit (50km/h): 0.00000
* General caution: 0.00000
* Speed limit (80km/h): 0.00000

The second image
* Right-of-way at the next intersection: 1.00000
* Beware of ice/snow: 0.00000
* Double curve: 0.00000
* Pedestrians: 0.00000
* End of no passing by vehicles over 3.5 metric tons: 0.00000

The third image
* Priority road: 1.00000
* End of all speed and passing limits: 0.00000
* Yield: 0.00000
* Keep right: 0.00000
* End of no passing: 0.00000

The fourth image
* Vehicles over 3.5 metric tons prohibited: 1.00000
* No passing: 0.00000
* Speed limit (100km/h): 0.00000
* End of no passing by vehicles over 3.5 metric tons: 0.00000
* Speed limit (60km/h): 0.00000

The fifth image
* General caution: 1.00000
* Traffic signals: 0.00000
* Pedestrians: 0.00000
* Road narrows on the right: 0.00000
* Road work: 0.00000

The sixth image
* Turn right ahead: 1.00000
* Ahead only: 0.00000
* Go straight or right: 0.00000
* Go straight or left: 0.00000
* Turn left ahead: 0.00000

The seventh image
* Keep right: 1.00000
* Turn left ahead: 0.00000
* Roundabout mandatory: 0.00000
* Speed limit (50km/h): 0.00000
* Speed limit (20km/h): 0.00000

The eighth image
* Keep right: 1.00000
* Turn left ahead: 0.00000
* Roundabout mandatory: 0.00000
* Speed limit (50km/h): 0.00000
* Speed limit (20km/h): 0.00000

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Below is the sample of original image.

![alt text][ahead-only]

Below is the output from the first convolutional layer
![alt text][conv1]

Lastly, the second convolutional layer output can be visualized as the image below.
![alt text][conv2]