# Unscented Kalman Filter Project
This is the second project of term 2 of Udacity's Self-Driving Car Nanodegree.

An unscented Kalman filter is a way to get around the problems that arise in an extended Kalman filter because of the linear approximation used by the extended filter. The idea goes as follows:

1. Make a list of points that are spread evenly across the extent of the measurement space using your current mean and variance to plot them.

2. Transform those points to where you think they will be the in the next measurement.

3. Find the new mean and variance of the new points

4. Use that to figure out how accurate your estimation and measurement are.

---

## Dependencies

* cmake >= v3.5
* make >= v4.1
* gcc/g++ >= v5.4

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./UnscentedKF path/to/input.txt path/to/output.txt`. You can find
   some sample inputs in 'data/'.
    - eg. `./UnscentedKF ../data/sample-laser-radar-measurement-data-1.txt output.txt`

## Generating Additional Data

If you'd like to generate your own radar and lidar data, see the
[utilities repo](https://github.com/udacity/CarND-Mercedes-SF-Utilities) for
Matlab scripts that can generate additional data.
