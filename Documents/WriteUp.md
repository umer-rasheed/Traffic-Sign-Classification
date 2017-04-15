#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.


###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the 2nd-3rd code cell of the IPython notebook.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 39209
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 4th-6th code cell of the IPython notebook. 

Exploratory visualization of the data set is shown in 4th-5th code cell. Bar chart showing the data is illustrated in 6th code cell 



###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in 7th code cell.
Initially I converted to Grayscale image but the results did not improve. Hence I used the RGB data.
I resample the data. The original data may be in some special order. If only part of the ordered-data are trained, then some kinds of classes may never be trained.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the 8th code cell of the IPython notebook. 
Currently, additional data is not generated.
I used "train_test_split()" from sklearn.cross_validation module. Only 5 percent of the original training data was taken as validation data. 
* The size of training set is 37248
* The size of validation set is 1961


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 11th cell of the ipython notebook. 

My final model consisted of the following layers:
Layer 1: Convolutional.
	Input=32x32x3. 
    	Output=28x28x50.
	Activation. ReLU.
	Pooling. The output shape should be 14x14x50.
Layer 2: Convolutional.
	Input=14x14x50. 
    	Output=10x10x80.
	Activation. ReLU.
	Pooling. The output shape should be 5x5x80.
Layer 3: Fully Connected Network.
	Input=5x5x80 (Flattened). 
    	Output=120.
Layer 3: Fully Connected Network.
	Input=120. 
    	Output=80.
Layer 3: Fully Connected Network.
	Input=80 . 
    	Output=43.
		

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 12th cell of the ipython notebook. 
Training set is shuffled before each training Epoch. 
Epoch number is set as 20
Batch Size is set as 128
I used Cross entropy function with softmax as the cost function.
Kingma and Ba's Adam algorithm is used as the optimizer. 
Learning rate is set as 0.001
Iteratively during each Epoch, accuracy is computed via validation set.


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 15th & 18th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 
* validation set accuracy of 97.8
* test set accuracy of 90.7

First Trial Process description
1). LeNet-5 neural network architecture. In the first convolutional layer, 6 features (classes) were used. Th the 2nd convolutional layer, 16 features were used;
2). Grayscale
3). Histogram Equalization using Clahe API in OpenCV


Second Trial Process description
1). LeNet-5 neural network architecture. In the first convolutional layer, 30 features (classes) were used. Th the 2nd convolutional layer, 50 features were used;
2). Grayscale
3). Histogram Equalization using Clahe API in OpenCV
Conclusion
Overall accuracy was improved

Third Trial Process description
1). LeNet-5 neural network architecture. In the first convolutional layer, 30 features (classes) were used. Th the 2nd convolutional layer, 50 features were used;
2). Luminance was used instead of Grayscale. Computed by converted BGR to YUV and then using the Y component (Luma).
3). Histogram Equalization using Clahe API in OpenCV
Conclusion
Overall accuracy was improved slightly

Fourth Trial Process description
1). LeNet-5 neural network architecture. In the first convolutional layer, 30 features (classes) were used. Th the 2nd convolutional layer, 50 features were used;
2). RGB was used

Fifth Trial Process description
1). LeNet-5 neural network architecture. In the first convolutional layer, 50 features (classes) were used. Th the 2nd convolutional layer, 80 features were used;
2). RGB was used
Conclusion:
Overall accuracy improved
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 86-114 cell of the Ipython notebook.
Compared to the training image, the new image has different sign size. And there is change in illumination.						
The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. This means that the parameters are specifically trained for the given traffic signs classification dataset. 
The results might change due to changes in weather, illumination, shape etc.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 114th cell of the Ipython notebook.
The first image yields incorrect prediction. However, the last 5 new testing images are predicted correctly. 
The first image, the model provided almost similar probabilities for "Speed Limit (30 km/h)", "Speed Limit (50 km/h)", "Speed Limit (60 km/h)". This may be due to the fact that '6' in the test image is somehwat different as compared to the one in training dataset. The rest of the images show that the probability of the correctly predicted class is significantly higher than the rest. 


