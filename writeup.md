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

[image1]: ./writep_images/traningdataimages.JPG "Visualization"
[image2]: ./writep_images/histogram.jpg "Histogram"
[image3]: ./writep_images/nomalization.jpg "Normalization"
[image4]: ./writep_images/accuracygraph.jpg "Accuracy"
[image5]: ./test_images/1.png "No passing"
[image6]: ./test_images/2.png "Speed limit 80km/h"
[image7]: ./test_images/3.png "Speed limit 60km/h"
[image8]: ./test_images/4.png "Bicycles Crossing"
[image9]: ./test_images/5.png "Speed limit 50km/h"
[image10]: ./test_images/6.png "No entry"
[image11]: ./test_images/prediction1.jpg "Prediction1"
[image12]: ./test_images/prediction2.jpg "Prediction2"
[image13]: ./test_images/prediction3.jpg "Prediction3"
[image14]: ./test_images/prediction4.jpg "Prediction4"
[image15]: ./test_images/prediction5.jpg "Prediction5"
[image16]: ./test_images/prediction6.jpg "Prediction6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 104397
* The size of the validation set is 441043
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to use the images in rgb, because in some case the colour could be important in traffic signals, but however i made a image normalization process because normalization is useful to have a certain independence of image properties, such as brightness and contrast

Here is an example of a traffic sign image before and after normalization.

![alt text][image3]

I decided to generate additional data because ... 

To add more data to the the data set, I used the technique of rotating 10 for each side and then add it to the train dataset

The difference between the original data set and the augmented data set is 313191-104397= 208.794


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x4 					|
| Convolution 5x5	    | 1x1 stride,  outputs 10x10x16					|
| RELU					|  												|
| Max pooling			| 2x2 stride,  outputs 5x5x16 					|
| Flatten				| Output 400									|
| Fully connected		| Imput 400 Output 120							|
| RELU					|  												|
| Fully connected		| Imput 120 Output 84							|
| RELU					|  												|
| Fully connected		| Imput 84 Output 43							|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used a basic LeNet model. I used the Adam Optimizer, the learning rate was 0.001, for 25 epochs with a batch size of 156. I adjusted the hiperparameters to obtain the grater accuracy without having fluctuences in accuracy

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training Accuracy = 0.993
* Validation Accuracy = 0.942
* Test Accuracy = 0.930

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?I only test with LeNet architecture because it provide me good results easily.
* What were some problems with the initial architecture? I didn't have a lot problems with this architecture.
* To adjust the LeNet CNN i searched on web some common starting values and started to adjust epochs and accuracy until the precision graphs give us values close to 1 and it stabilizes
* Which parameters were tuned? learning rate and epochs

If a well known architecture was chosen:
* What architecture was chosen? LeNet CNN
* Why did you believe it would be relevant to the traffic sign application? Because this CNN architecture is useful to detec diferent features in images.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? Te system is working very well beacuse the training accuracy (0,993) is quite close to validation accuracy (0.942) and test accuracy (0.930)



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] ![alt text][image10]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No passing     		| No passing  									| 
| Speed limit 80km/h	| Speed limit 80km/h							|
| Speed limit 60km/h	| Speed limit 60km/h							|
| Bicycles Crossing		| Bicycles Crossing 			 				|
| Speed limit 50km/h	| Speed limit 50km/h							|
| No entry 				| No entry  									|

The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. T

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is sure that this is a "no passing" sign (probability of 100), and the image does contain a "no passing" sign . The top five probabilities were

![alt text][image11]

For the second image, the model is sure that this is a "Speed limit 80km/h" sign (probability of 100), and the image does contain a "Speed limit 80km/h" sign. The top five probabilities were

![alt text][image12]

For the third image, the model is relatively sure that this is a "Speed limit 60km/h" sign (probability of 74), and the image does contain a "Speed limit 60km/h" sign. The top five probabilities were

![alt text][image13]

For the fourth image, the model is found the correct image but in this case with less accuracy (probability of 55). The top five probabilities were

![alt text][image14]

For the second image, the model is sure that this is a "Speed limit 50km/h" sign (probability of 100), and the image does contain a "Speed limit 50km/h" sign. The top five probabilities were

![alt text][image15]

For the second image, the model is sure that this is a "No entry" sign (probability of 100), and the image does contain a "No entry" sign. The top five probabilities were

![alt text][image16]



