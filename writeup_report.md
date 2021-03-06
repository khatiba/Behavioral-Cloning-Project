# Self Driving Car - Behavioral Cloning Project 


### Introduction

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results below

Udacity provides a [car simulator](https://github.com/udacity/self-driving-car-sim) to collect training data and run the model.
The simulator has 2 tracks and both are used to train and test. The output of the simulator is a CSV file with image paths to the 
center, left and right cameras mounted on the car. The CSV also has telemetry including stearing angle, throttle and speed.

The input data is used in NVIDIA's CNN shown below and the output model is used to drive the car in the simulator.

### Setup

* `model.py` or `Behavioral-Cloning-Project.ipynb` contains the code for training and saving the output of the CNN. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Notice that just out of convenience, the model assumes two folders of data `track-1` and `track-2`, I keep the data separated by track in case I mess up a recording run.

* `drive.py` is used to run the model in the simulator, I used this to try different controller settings and speeds.

* `model.h5` contains all of the weights of the model.

* `video.mp4` [video of autonomous](./video.mp4) run on track 1.

### Run the simulation

First setup your environment with the udacity term-1 starter kit and also install Keras 2.x. The model was trained on the latest version of Keras, but the starter kit uses 1.x.

You will need to clone the repo, update the Keras version and build a new image.

`$ git clone https://github.com/udacity/CarND-Term1-Starter-Kit.git`


Update the `environment.yml` and `environment-gpu.yml` at the bottom `keras==1.2.1` -> `keras==2.0.6`

Build yourself a new Docker image:

`$ docker build -t my-new-carnd-term1-starter-kit .`


And now you can run the model and simulator:

``
`docker run -it --rm -p 4567:4567 -v /absolute/path/to/this/repo/:/src my-new-carnd-term1-starter-kit python drive.py model.h5`
``

Load up the simulator, select the first track and click `Autonomous Mode`


### Model Architecture and Training Strategy 

[//]: # (Image References)

[image1]: ./examples/model-architecture.png "Model Visualization"
[image2]: ./examples/error-visualization.png "Error Visualization"
[image3]: ./examples/train-recovery.gif "Recovery Image"
[image4]: ./examples/autonomous-run-close-call.gif "Autonomous Recovery Image"
[image6]: ./examples/normal-image.png "Normal Image"
[image7]: ./examples/processed-image.png "Processed Image"
[image8]: ./examples/flipped-image.png "Flipped Image"


#### 1. Model Layers and Activation

My model starts with the NVIDIA CNN shown in the figure below.

There are two additional layers at the start (not shown in the diagram) that normalize and crop the input images.

The model includes RELU activations after every convolution to introduce nonlinearity (code line 97-101), and the data is normalized in 
the model using a Keras lambda layer (code line 94) and cropped (code line 95).

I modified the network at the end by adding a dropout layer between the first two fully connected layers on line 104.

Because the data set is so large, it cannot all be fit in memory at once. Python generators are used to extract batches of data to read 
and preprocess before sending them to the CNN, this is on line 58 `def batch_generator(samples):`.

![alt text][image1]


#### 2. Steps to Reduce Overfitting

The model contains a dropout layer in order to reduce overfitting (model.py lines 104).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 39-45).
 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I also augmented the data set by flipping all of the images from left to right and inverting the stearing angle measurement, 
this doubled the data set but also prevents the network from biasing one direction (code lines 76-85).

In addition to the normalization I also added a slight Gaussian blur to all of the images.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 111).

#### 4. Appropriate training data

Training data was carried out by driving in the center of the road on both tracks. I used a combination of center lane driving and 
recordings of recovering from the left and right sides of the road to train the network how to correct itself.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with previously known networks such as LeNet. After some trial and error, 
the NVIDIA CNN was used since it has been designed specifically for self driving cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that 
many of my trials had low mean squared error (MSE) on the training set but a high error on the validation set, implying that the model was overfitting.

To combat the overfitting, I modified the model by adding dropout before the first fully connected layer. 

After more trials, I had the same results, low MSE on training but high on validation. So then I turned to the training data itself. 
I gathered more and more data points but still only had small improvements.

After scanning the CSV, I noticed that there were mostly zeros in the stearing angle column, this is because I was driving the car with the 
arrow keys which cause either very high or zero stearing angles, to guide the car it required lots of tapping of the keys resulting in noisy measurements.

I decided to recapture all the data again but use the mouse instead to get better angle measurements. This did the trick and I achieved great 
results with both MSE values being very low and close to each other.

The final step was to run the simulator to see how well the car was driving around track one. At the end of the process, the vehicle is able to 
drive autonomously around the track without leaving the road, and did a pretty good job of staying centered as well.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

* Lambda layer to normalize
* Copping layer
* Convolution 5x5 filter, 24 deep
* Convolution 5x5 filter, 36 deep
* Convolution 5x5 filter, 48 deep
* Convolution 3x3 filter, 64 deep
* Convolution 3x3 filter, 64 deep
* Flattened followed by Dropout at 50%
* Fully connected 100
* Fully connected 50
* Fully connected 10
* Final single stearing angle output


Here is a visualization of the errors as the model trains:

![alt text][error-visualization]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on each track going opposite directions. I then took snapshot recordings of the 
car veering off the road and recoving back to center. I also did multiple runs through corners at slow speeds to get very smooth stearing angle measurments.

![alt text][image2]

I then recorded the vehicle recovering from the left and right side of the road back to center so that the vehicle would learn to correct 
it self if it swayed towards the edges. These images show what a recovery looks like:

![alt text][image3]

During autonomous driving, here is an example of a close call, but nice recovery:

![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles so that the model wouldn't bias one direction.

![alt text][image6]

After normalizing, blurring and cropping:

![alt text][image7]
![alt text][image8]


After the collection process, I had close to 40,000 data points and with the flipped images, the total dataset was about 80,000 with 25% extracted as a validation set.

Initially I forgot to shuffle the data, this led to odd results, training error would decrese during the first half of an epoch, then climb towrads the end,
training MSE was still low, but validation MSE was high, seeing this was a sign I didn't shuffle the samples.

After shuffling, I split the samples 75% for training and 25% for validation.

I then preprocessed this data by applying a small Gaussian blur, cropping the top and bottom of the image, converting the color space to YUV and normalizing.

The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by watching as loss reached a 
minimum and then began increasing. I used an adam optimizer so that manually training the learning rate wasn't necessary.

```
Epoch 8/15
58/58 [==============================] - 126s - loss: 0.0134 - val_loss: 0.0130
Epoch 9/15
58/58 [==============================] - 126s - loss: 0.0127 - val_loss: 0.0121
Epoch 10/15
58/58 [==============================] - 126s - loss: 0.0121 - val_loss: 0.0122
Epoch 11/15
58/58 [==============================] - 126s - loss: 0.0125 - val_loss: 0.0127
Epoch 12/15
58/58 [==============================] - 126s - loss: 0.0132 - val_loss: 0.0131
```


### Conclusion

The CNN model worked very well after working out some issues with the training data. However, there is still much room for improvement, as seen in the video
the car sways slightly from side to side and had a near miss on one of the turns. Some of this can be attributed to quality of the training runs.
I found that many of the issues I faced were related to the training data, I've listed some key areas to note:

* Take care in gathering the training data and ensure it;s accurate as possible (use the mouse to stear, measurements are much better!)
* Shuffle the data set, a sign that I forgot to do this was seeing an initial decrease in training error than a rapid increase
* Record the tracks going both ways to avoid biases and help prevent the model from memorizing the track
* Re-capture difficult areas or turns
* Augment the data by flipping the images to ensure no bias is learned
* Normalize and crop to focus on the important information and improve performance
* Add dropout - this has been very effective at reducing overfitting and it was the first thing I added to the model

**Further development:**

* Try more dropout, more preprocessing, grayscaling and randomly distort the images
* Gather more training data and test the car on a new track not seen yet
