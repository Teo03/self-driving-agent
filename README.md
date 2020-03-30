
# AI Self Driving Carla Agent

  

Inspired by the [DeepPiCar](https://towardsdatascience.com/tagged/deep-pi-car) series I've tried to implement Nvidia's End to End Learning for Self-Driving Cars [Paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) in the Carla Simulator. The agent learned to predict the **steering angles** based on the inputted images from only the front camera sensor attached to the agent. The data was collected manually by driving the car around the map without taking a lot of extreme turns (>0.5).

Trained on about 3k images and their according angles the car can perform soft turns and follow the driving lane mostly without crashing.

  

## Prerequisites

You should have the [latest Carla version](https://github.com/carla-simulator/carla/releases) installed and exported the ***WindowsNoEditor*** folder one directory above the ***self-drive*** folder.

  

You should have TensorFlow, Keras, Pygame, OpenCV, scikit-learn, and PyGame installed. I have only tested it on Windows 10, but it should work on Linux.

  

## Collecting training data

To train with your own data you can generate data either manually or by using the built-in Carla Autopilot (this can result in taking extreme turns or duplicate data). You must be inside the *self-drive* folder.

  

To record using Carla Autopilot.

  

    > python client.py collect [how many seconds to run for]

To record manually enter the following command and after loading press ***Backspace*** to start recording.

  

    > python manual_control.py

  

This will create a *generated_data* folder with ***image_data/*** folder and ***data.csv*** file containing the image names and the steering angles accordingly.

  

## Training the model

To train the model you must be inside the *self-drive* folder and specify the *path* for storing the model and the *path* of your saved training data.

  

    > python client.py train [path to data] [path to where you want the model to be saved]

  

You can optionally start a TensorBoard server.

    tensorboard --logdir=logs

The default training parameters can be tweaked in `client.py` .
  

## Running

After the training, you can run the models by using the pre-trained models located inside the ***models/*** folder or by replacing them with your own models. For best performance set the ***-quality-level=Low*** when starting the Carla server with the ***startServer.bat*** file (Windows).

  

    > python client.py drive [how many seconds to drive before destroying actors]

  
  

Feel free to contribute or open a pull request!