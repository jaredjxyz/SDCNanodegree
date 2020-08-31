# CarND-Controls-PID

This is the fourth project in term 2 of Udacity's Self-Driving Car Nanodegree. The goal is to make a PID controller, which makes a car steer toward and follow along with a line. A PID controller has 3 parts:

1. P: Proportional. This just means that if the car is further away from the center line, it's going to turn more toward the center line.

2. I: Integral. This just means that it's going to take into account if it's been mostly on the right side or mostly on the left side (as it wants to spend approximately the same amount of time on both sides). If the robot has been mostly on the left side, it's going to turn more right. The integral is there to counteract any steering misalignments.

3. D: Derivative. This just means that if the car isn't facing the same direction as the line it wants to follow, then it's going to turn such that it will eventually be going paralell to the line.

These three terms together make a car follow a line smoothly, and the only information a car needs to know to make it work is how far away the car is from the center line at any given point.

---

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets) == 0.13, but the master branch will probably work just fine
  * Follow the instructions in the [uWebSockets README](https://github.com/uWebSockets/uWebSockets/blob/master/README.md) to get setup for your platform. You can download the zip of the appropriate version from the [releases page](https://github.com/uWebSockets/uWebSockets/releases). Here's a link to the [v0.13 zip](https://github.com/uWebSockets/uWebSockets/archive/v0.13.0.zip).
  * If you run OSX and have homebrew installed you can just run the ./install-mac.sh script to install this
* Simulator (Tested on v1.1). You can download these from the [project intro page](https://github.com/udacity/CarND-PID-Control-Project/releases) in the classroom.

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Download the [simulator](https://github.com/udacity/CarND-PID-Control-Project/releases) and run it on "Autonomous" mode.
5. Run the PID controller: `./pid`. 

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!
