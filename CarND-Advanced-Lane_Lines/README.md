# Advanced Lane Finding

The objective of this project was to expand on the original Finding Lane Lines project in order to improve the lines we found. We were give five sample images and a sample video, and told to use various computer vision techniques to find each of the lines on the road. Here's my solution.

## TODO

* The program works really well on the test video, but does not extend well to other videos. I'd like to continue working on this to find a threshold that works for more videos, implement some dynamic thresholding, or use some a support vector machine to find the best line instead of doing it with computer vision techniques

## Discussion

The biggest problem I had in making this was finding thresholds that worked well. This is where my project is most likely to fail. The thresholds right now are set to very specific values to cater to this specific road and amount of daylight. I've tried it on other videos and, unfortunately, it does not generalize well. Another place where it may fail is in finding lane lines. Right now it depends on finding one line really well, then using the data from the other line to augment that data and find a nice balance between the two. If it doesn't find either line very well, it could potentially draw the lines all over the place. If I were going to continue the project, I would find a way to adjust the thresholds based off the current image so that it could adjust to new images. Perhaps continue changing the threshold until a certain number of pixels are activated?

## Videos

<a href="https://youtu.be/6B6QawlZBgI" target="_blank"><img src="http://img.youtube.com/vi/6B6QawlZBgI/0.jpg" width="360" /></a> <a href="https://youtu.be/LyacTp2JgEk" target="_blank"><img src="http://img.youtube.com/vi/LyacTp2JgEk/0.jpg" width="360" /></a>

## Pipeline `find_lane_lines.pipeline`


Here are the steps taken to get the lines from the following test images. You can find the code for this inside of

<img src="output_images/test_images/test1_original.jpg" width="360"/> <img src="output_images/test_images/test2_original.jpg" width="360"/> <img src="output_images/test_images/test3_original.jpg" width="360"/> <img src="output_images/test_images/test4_original.jpg" width="360"/> <img src="output_images/test_images/test5_original.jpg" width="360"/> <img src="output_images/test_images/test6_original.jpg" width="360"/>

### Undistorting images `find_lane_lines.undistort` `find_lane_lines.calibrate_camera`

An image taken by a camera tends to be distorted by the lens it uses. This can be seen exaggerated in a fish-eye lens, but any lens is going to have some sort of distortion. Fortunately, OpenCV has some really useful tools to correct for this distortion. I took a few distorted images of a chessboard and used OpenCV's 
ability to find the corners inside of the chessboard for each image. Using OpenCV, we can easily take those photos and calculate how the camera is distorting the image, then undistort it.

<img src="camera_cal/calibration1.jpg" width="360"/> <img src="output_images/camera_cal/calibration1.jpg" width="360" />

<img src="output_images/test_images/test1_undis.jpg" width="360"/> <img src="output_images/test_images/test2_undis.jpg" width="360"/> <img src="output_images/test_images/test3_undis.jpg" width="360"/> <img src="output_images/test_images/test4_undis.jpg" width="360"/> <img src="output_images/test_images/test5_undis.jpg" width="360"/> <img src="output_images/test_images/test6_undis.jpg" width="360"/>

### Thresholding `find_lane_lines.apply_threshold`

We can convert the image into different "Color spaces", then choose which values for each pixel we want to keep.

I used the following methods:

1. Sobel in the x direction.

   Sobel is an edge decection that works by finding how much a color changes between two pixels. It returns an image with all of the vertical edges highlighted. This is good for finding vertical lane lines, but also picks up all of the other vertical edges.

2. HLS color space conversion.

   HLS is a color space like RGB, except that instead of storing Red, Green, and Blue values, it stores Hue, Lightness, and Saturation. The nice thing about this is that in an image where there are shadows, the only value that changes is the Lightness value, where the Hue and Saturation values stay the same. This allows one to find continuous lane lines across shadows.

I then took my three color channels (Sobel, Hue, and Saturation) and experimented with finding a good balance of color calues to choose between them. I then set the selected pixels' color values to 255, and set the unselected pixels' values to 0 for each threshold. From there, for each pixel, the program looks at the three thresholds. If the pixel is activated in 2 of the 3 thresholds, it's selected for the final threshold.

<img src="output_images/test_images/test1_thresholded.jpg" width="360"/> <img src="output_images/test_images/test2_thresholded.jpg" width="360"/> <img src="output_images/test_images/test3_thresholded.jpg" width="360"/> <img src="output_images/test_images/test4_thresholded.jpg" width="360"/> <img src="output_images/test_images/test5_thresholded.jpg" width="360"/> <img src="output_images/test_images/test6_thresholded.jpg" width="360"/>

### Perspective Transform `find_lane_lines.warp`

Though in these pictures it's easy to see where the lines are, it's not trivial to figure out the angle at which the lines are curving. To make this easier, we take the image and transform it so that it looks like we're looking at it from the sky instead from the ground. From this perspective, it's easier to see how much the road is curving.

<img src="output_images/test_images/test1_warped_thresholded.jpg" width="360"/> <img src="output_images/test_images/test2_warped_thresholded.jpg" width="360"/> <img src="output_images/test_images/test3_warped_thresholded.jpg" width="360"/> <img src="output_images/test_images/test4_warped_thresholded.jpg" width="360"/> <img src="output_images/test_images/test5_warped_thresholded.jpg" width="360"/> <img src="output_images/test_images/test6_warped_thresholded.jpg" width="360"/> 

### Calculate lines `find_lane_lines.get_lines_from_thresholded_image`

Now that we have possible line locations singled out, we need to plot the lines.

We first find the x coordinate in each half of the image with the most activated y coordinates. These will be our starting positions for each of the two lines we want to draw.

We then start at the bottom of the image and look at the first 1/10th of the height of the image. We find pixels within a margin of 100px of our base. Those pixels are added to our line. If there are more than 50 pixels in this section, we then re-position the current section to the location of their mean x position, and then continue up the picture.

After we've figured out which pixels we're going to count as part of our line, we find the best-fit quadratic line between those pixels. We then find a curvature that's somewhere between what each line wants, with a line that has more selected pixels getting more weight in the decision of what the final curvature will be. We also set their constant values to be approximately 480 pixels apart, using a log function to let them be slighly further away or closer together than 480 pixels.

This creates a relatively consistent line that matches up with a picture, but in a continuous video the line skips around quite a bit. We need to smooth out the line for a video. , and add that line to a queue of maximum 5 lines. We then choose the mean line of those five as our line.

<img src="output_images/test_images/test1_warped_thresholded_with_lines.jpg" width="360"/> <img src="output_images/test_images/test2_warped_thresholded_with_lines.jpg" width="360"/> <img src="output_images/test_images/test3_warped_thresholded_with_lines.jpg" width="360"/> <img src="output_images/test_images/test4_warped_thresholded_with_lines.jpg" width="360"/> <img src="output_images/test_images/test5_warped_thresholded_with_lines.jpg" width="360"/> <img src="output_images/test_images/test6_warped_thresholded_with_lines.jpg" width="360"/> 

### Undo Perspective Transform `find_lane_lines.unwarp`

We then take this new image and put it back into the perspective that we started with.

<img src="output_images/test_images/test1_thresholded_with_lines.jpg" width="360"/> <img src="output_images/test_images/test2_thresholded_with_lines.jpg" width="360"/> <img src="output_images/test_images/test3_thresholded_with_lines.jpg" width="360"/> <img src="output_images/test_images/test4_thresholded_with_lines.jpg" width="360"/> <img src="output_images/test_images/test5_thresholded_with_lines.jpg" width="360"/> <img src="output_images/test_images/test6_thresholded_with_lines.jpg" width="360"/> 

### Final Result `find_lane_lines.pipeline`

<img src="output_images/test_images/test1_with_lines.jpg" width="360"/> <img src="output_images/test_images/test2_with_lines.jpg" width="360"/> <img src="output_images/test_images/test3_with_lines.jpg" width="360"/> <img src="output_images/test_images/test4_with_lines.jpg" width="360"/> <img src="output_images/test_images/test5_with_lines.jpg" width="360"/> <img src="output_images/test_images/test6_with_lines.jpg" width="360"/> 
