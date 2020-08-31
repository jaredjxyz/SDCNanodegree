# Udacity Self-Driving Car Behavioral Cloning Project

## Introduction

This was one of the coolest projects I've ever done. Udacity has an [Open Source Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) which allows you to drive around a car on a virtual track, record the images and driving data (steering angle, braking data, and throttle) for each frame, and then allows you to make a program to take in images and send back driving data in real time to drive the car around.

The goal of the project was to make neural network that drives the car around the track using a neural network. I not only succeeded in doing this, but also succeeded in making a model that drives around the track at full speed the whole time!

Please see my [write-up](writeup_report.pdf) for further explanation.

## Getting Started

**UPDATE, March 28 2017**: Unfortunately, an update to Udacity's simulator program broke the model some time around late February, 2017. As of right now, the model does not work reliably any publicly available version, and has decided that it likes to run off the road at the beginning when going full speed, and while going slowly it likes to run all over the track (though it tends to stay on the track). I intend to fix this, but in the meantime, check out the videos.

Download Udacity's [Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim). This is only tested in version 1 of the simulator.

Make sure you have [Git LFS](https://git-lfs.github.com/) installed before cloning the repository

Install OpenCV, Keras, and SKLearn Python modules.

### Running the model around the track:

Open the car simulator, and choose a graphics setting (The model works best on the lowest graphics setting because that's the setting with the lowest latency).

Select the track on the left (The one on the right didn't exist when I made this) and click "Autonomous Mode"

Watch as the car drives itself!

### Training the model

Warning: This uses a lot of memory and resources. I highly recommend using tensorflow-GPU for this.

Unpack the training data:

`tar -xzf data_final.tgz`

then run the training program:
`python model.py`

## Videos

Here are full throttle and 1/5 throttle videos of the model running.

[![Full Speed](https://img.youtube.com/vi/hflPGI8BXa4/0.jpg)](https://www.youtube.com/watch?v=hflPGI8BXa4)
[![1/5 Speed](https://img.youtube.com/vi/Et-LhJ4XKW0/0.jpg)](https://www.youtube.com/watch?v=Et-LhJ4XKW0)
