import argparse
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import glob
import scipy.stats
from collections import deque


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_calibration_directory', default='camera_cal/')
    parser.add_argument('--test_directory', default='test_images/')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save_directory', default='output_images/')
    parser.add_argument('--video', nargs='?', const='project_video.mp4', default=None)

    args = parser.parse_args()
    camera_settings = calibrate_camera(args.camera_calibration_directory)

    if args.video:
        cap = cv2.VideoCapture(args.video)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(args.save_directory + args.video, fourcc, fps, size)

        size_tiled = (size[0] * 2, size[1] * 2)
        out_tiled = cv2.VideoWriter(args.save_directory + 'tiled_' + args.video, fourcc, fps, size_tiled)
        i = 0
        while cap.isOpened():

            # Read in the next frame
            ret, frame = cap.read()

            if not ret:
                break

            i += 1
            print(str(i) + '/' + str(num_frames))
            # if i < 500:
            #     continue

            pipelined_image, extracted_images = pipeline(frame, camera_settings, extract_images=True, continuous=True)
            tiled_image = tile_images(((extracted_images['original'],
                                        extracted_images['with_lines_and_words']),
                                       (cv2.cvtColor(extracted_images['warped_thresholded'], cv2.COLOR_GRAY2BGR),
                                        extracted_images['warped_thresholded_with_lines'])))

            out.write(pipelined_image)
            out_tiled.write(tiled_image)
            del extracted_images
            # cv2.imshow('frame', pipelined_image)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    else:
        # Save all images
        image_locations = glob.glob(args.test_directory + '*.jpg')
        for image_name in image_locations:
            image = cv2.imread(image_name)
            pipelined_image, extracted_images = pipeline(image, camera_settings, extract_images=True, continuous=False)

            if args.save:
                # Save checkerboard images
                checkerboard_image_names = glob.glob(args.camera_calibration_directory + '*.jpg')
                for checkerboard_path in checkerboard_image_names:
                    dirname = os.path.dirname(checkerboard_path)
                    if not os.path.exists(args.save_directory + dirname):
                        os.makedirs(args.save_directory + dirname)
                    undis_checkerboard = undistort(cv2.imread(checkerboard_path), camera_settings)
                    cv2.imwrite(args.save_directory + checkerboard_path, undis_checkerboard)

                # Save lane distances
                basename = os.path.basename(image_name)
                name, extension = basename.split('.')
                dirname = os.path.dirname(image_name)

                if not os.path.exists(args.save_directory + dirname):
                    os.makedirs(args.save_directory + dirname)
                for suffix in extracted_images:
                    cv2.imwrite(args.save_directory + dirname + '/' + name + '_' + suffix + '.' + extension, extracted_images[suffix])
            else:
                show_image(pipelined_image)


def calibrate_camera(calibration_image_directory, corners_shape=(9, 6)):
    '''
    Calibrates the camera, given the directory of a bunch of jpegs of chessboards
    and the number of inside corers in that chessboard
    '''
    nx, ny = corners_shape

    # Read calibration images

    objpoints = []  # 3D points in real space
    imgpoints = []  # 2d points on chessboard

    # Set where corners should be mapped to. (0,0,0)--(nx-1,ny-1,0)
    mappedPoints = np.zeros((nx * ny, 3), np.float32)
    mappedPoints[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Load images into arrays
    for image_name in glob.glob(calibration_image_directory + '*.jpg'):

        image = cv2.imread(image_name, 0)
        ret, corners = cv2.findChessboardCorners(image, corners_shape, None)
        if ret:
            objpoints.append(mappedPoints)
            imgpoints.append(corners)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image.shape[::-1], None, None)
    if ret:
        return camera_matrix, dist_coeffs


def warp(image):
    '''
    Warps the image to a top-down view
    '''

    source_points = np.float32([[220, 700], [1060, 700], [595, 450], [685, 450]])
    dest_points = np.float32([[400, 720], [880, 720], [400, 0], [880, 0]])

    M = cv2.getPerspectiveTransform(source_points, dest_points)

    return cv2.warpPerspective(image, M, image.shape[:2][::-1], flags=cv2.INTER_LINEAR).astype(np.uint8)


def unwarp(image):
    '''
    Warps the image from top-down view back to original
    '''
    source_points = np.float32([[220, 700], [1060, 700], [595, 450], [685, 450]])
    dest_points = np.float32([[400, 720], [880, 720], [400, 0], [880, 0]])

    M = cv2.getPerspectiveTransform(dest_points, source_points)

    unwarp = cv2.warpPerspective(image, M, image.shape[:2][::-1], flags=cv2.INTER_LINEAR)
    return unwarp


def show_image(image, title='preview'):
    '''
    Displays an image on the screen
    '''
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_image(image):
    '''
    Displays the image, plotted, on the screen
    '''
    plt.imshow(image, cmap='gray')
    plt.show()


def sobel(image, orient):
    '''
    Applies the sobel operation to the image, in the 'x' or 'y' orientations
    '''
    if orient == 'x':
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    abs_sobel = np.abs(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    return scaled_sobel


def threshold_binary(image, thresh_min=0, thresh_max=255):
    '''
    Applies a binary threshold to the given image
    '''
    mask = np.zeros_like(image)
    mask[(image >= thresh_min) & (image <= thresh_max)] = 255
    return mask


def apply_threshold(image,
                    sobel_threshold=(10, 100),
                    saturation_threshold=(100, 255),
                    hue_threshold=(200, 255),
                    continuous=False):
    '''
    Does the full task of doing all thresholds to the image
    '''

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = sobel(gray, 'x')
    sobelx = sobel(sobelx, 'x')
    sobelx = cv2.blur(sobelx, (3, 3))

    sobel_thresh = threshold_binary(sobelx, *sobel_threshold)

    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:, :, 2]
    saturation_thresh = threshold_binary(saturation, *saturation_threshold)

    hue = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:, :, 0]
    hue_thresh = threshold_binary(hue, *hue_threshold)

    # warped_combination = warp(sobel_thresh & saturation_thresh & hue_thresh)
    best2of3 = (sobel_thresh / 255 + saturation_thresh / 255 + hue_thresh / 255) >= 2
    combination = best2of3.astype(np.float32) * 255
    # show_image(warped_combination)

    return combination


right_lines = deque(maxlen=3)
left_lines = deque(maxlen=3)


def get_lines_from_thresholded_image(img, continuous=False):
    '''
    Takes in a thresholded image and draws lines on it, then returns
    the image and the set of lines
    '''
    histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)
    midpoint = np.int(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    n_windows = 10
    window_height = np.int(img.shape[0] / n_windows)
    margin = 100
    minpix = 50

    # List of nonzero indices
    nonzero_y, nonzero_x = img.nonzero()

    left_base_current = left_base
    right_base_current = right_base

    left_lane_inds = []
    right_lane_inds = []
    left_lane_weights = []
    right_lane_weights = []

    out_image = np.dstack((img, img, img)).astype(np.uint8) * 255

    for window in range(n_windows):
        # Get the bounds for the window
        y_top = img.shape[0] - (window + 1) * window_height
        y_bottom = img.shape[0] - window * window_height
        x_left_low = left_base_current - margin
        x_left_high = left_base_current + margin
        x_right_low = right_base_current - margin
        x_right_high = right_base_current + margin

        # Draw left rectangle
        cv2.rectangle(out_image, (x_left_low, y_bottom), (x_left_high, y_top), 2)
        # Draw right rectangle
        cv2.rectangle(out_image, (x_right_low, y_bottom), (x_right_high, y_top), 2)

        # Find the indices that are in the range we want
        good_left_inds = ((nonzero_y >= y_top) & (nonzero_y < y_bottom) & (nonzero_x >= x_left_low) & (nonzero_x < x_left_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= y_top) & (nonzero_y < y_bottom) & (nonzero_x >= x_right_low) & (nonzero_x < x_right_high)).nonzero()[0]

        # Find how powerful each position should be in the consideration
        right_weights = get_normal_weights(nonzero_x[good_right_inds], mean=right_base_current)
        left_weights = get_normal_weights(nonzero_x[good_left_inds], mean=left_base_current)

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        left_lane_weights.append(left_weights)
        right_lane_weights.append(right_weights)

        if len(good_left_inds) > minpix:
            left_base_current += np.int((np.mean(nonzero_x[good_left_inds]) - left_base_current) / 2)
        if len(good_right_inds) > minpix:
            right_base_current += np.int((np.mean(nonzero_x[good_right_inds]) - right_base_current) / 2)

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    left_lane_weights = np.concatenate(left_lane_weights)
    right_lane_weights = np.concatenate(right_lane_weights)

    leftx = nonzero_x[left_lane_inds]
    lefty = nonzero_y[left_lane_inds]
    rightx = nonzero_x[right_lane_inds]
    righty = nonzero_y[right_lane_inds]

    # Fits a polynomial of degree 2 to right and left
    left_fit = np.polyfit(lefty, leftx, 2, w=left_lane_weights)
    right_fit = np.polyfit(righty, rightx, 2, w=right_lane_weights)

    # Make the distance between fight and left lines approximately 480, with a little bit of wiggle room
    distanceBetween = right_fit[2] - left_fit[2]
    distanceChange = distanceBetween - 480
    distanceChange -= np.log(abs(distanceChange + 1))

    left_weight = len(leftx)**2 / (len(leftx)**2 + len(rightx)**2)
    right_weight = len(rightx)**2 / (len(leftx)**2 + len(rightx)**2)
    left_fit[2] += distanceChange * (1 - left_weight)
    right_fit[2] -= distanceChange * (1 - right_weight)

    # Set the curves equal, with the amount of pixels describing each curve being the amount that that curve contributes
    left_fit[1], right_fit[1] = [left_fit[1] * (left_weight) + right_fit[1] * (right_weight)] * 2
    left_fit[0], right_fit[0] = [left_fit[0] * (left_weight) + right_fit[0] * (right_weight)] * 2

    # Add these lines to the queue so that we can return the average
    if continuous:

        left_fit = np.mean(list(left_lines) + [left_fit], axis=0)
        right_fit = np.mean(list(right_lines) + [right_fit], axis=0)
        left_lines.append(left_fit)
        right_lines.append(right_fit)

    out_image = np.zeros_like(out_image)

    # Left lane is blue, right lane is green, best fit line is red
    # out_image[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
    # out_image[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 255, 0]
    out_image = draw_lane(out_image, (left_fit, right_fit), [0, 255, 0])

    # This is for adjusting for meters
    y_meters_per_pixel = 30 / 720
    x_meters_per_pixel = 3.7 / 480 * (700 / 720)

    left_fit_cr = np.polyfit(lefty * y_meters_per_pixel, leftx * x_meters_per_pixel, 2)
    right_fit_cr = np.polyfit(righty * y_meters_per_pixel, rightx * x_meters_per_pixel, 2)

    distanceBetween_cr = right_fit_cr[2] - left_fit_cr[2]
    distanceChange_cr = distanceBetween_cr - 480 * x_meters_per_pixel
    distanceChange_cr -= np.log(abs(distanceChange_cr + 1))

    y = (img.shape[1] - 1) * y_meters_per_pixel
    A, B, C = left_fit
    dx = 2 * A * y + B
    dx2 = 2 * A
    R = abs(1 + dx**2)**1.5 / abs(dx2)

    return (out_image, (left_fit, right_fit), R)


def draw_lane(img, lines, color=[0, 255, 0]):
    '''
    Takes in an image and a set of polynomial lines and plots the lines on the image
    '''
    out_image = np.copy(img)
    left_line, right_line = lines

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0]).astype(np.int)

    assert(len(lines) == 2)
    x_values = []
    for line in lines:
        # y indices

        degree = len(line)

        # x indices
        fitx = np.sum(line[i] * ploty**(degree - 1 - i) for i in range(degree)).astype(np.int)
        x_values.append(fitx)

        # Road in the middle is

    line1, line2 = x_values
    current_line = line1
    while all(current_line < line2):
        current_line += 1
        valid = np.where((0 <= current_line) & (current_line < img.shape[1]))
        out_image[ploty[valid], current_line[valid]] = color

    return out_image


def get_normal_weights(values, mean=None):
    '''
    Takes in a set of values, and returns an array of the same shape with the weights of those values
    '''

    if len(values) == 0:
        return values

    stdev = np.var(values)
    if mean is None:
        mean = np.mean(values)

    norm = scipy.stats.norm(mean, stdev).pdf(values)
    norm[np.isnan(norm)] = 1

    max_norm = max(norm)
    min_norm = min(norm)
    weights = (norm - min_norm) / (max_norm - min_norm)
    weights[np.isnan(weights)] = 1
    return norm


def undistort(image, camera_settings):
    '''
    Takes in a distorted image, pops out an undistorted image
    '''

    camera_matrix, dist_coeffs = camera_settings
    return cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)


def apply_threshold_to_picture(thresholded_image_with_lines, image):
    '''
    Lays a thresholded image on top of a picture
    '''

    lines_x, lines_y = np.where(((thresholded_image_with_lines[:, :, 0] != 255) |
                                 (thresholded_image_with_lines[:, :, 1] != 255) |
                                 (thresholded_image_with_lines[:, :, 2] != 255)) &
                                ((thresholded_image_with_lines[:, :, 0] != 0) |
                                 (thresholded_image_with_lines[:, :, 1] != 0) |
                                 (thresholded_image_with_lines[:, :, 2] != 0)))
    image_with_lines = np.copy(image)

    image_with_lines[lines_x, lines_y] = thresholded_image_with_lines[lines_x, lines_y]

    # Add transparency

    alpha = .5
    image_with_lines = cv2.addWeighted(image, alpha, image_with_lines, 1 - alpha, 0, image_with_lines)

    return image_with_lines


def write_statistics(image, lines, R):
    '''
    Writes statistics on the video
    '''
    center_of_image = image.shape[1] / 2
    center_of_car = (lines[0][2] + lines[1][2]) / 2
    meters_between_lines = 3.7 * (700 / 720)
    meters_per_pixel = meters_between_lines / 480
    car_offset_in_meters = (center_of_car - center_of_image) * meters_per_pixel

    image = np.copy(image)

    cv2.putText(image, "Curvature radius: " + str(R) + " meters", (0, int(image.shape[0] / 20)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_8)
    cv2.putText(image, "Distance from center: " + str(car_offset_in_meters) + " meters", (0, int(image.shape[0] / 10)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_8)
    return image


def pipeline(image, camera_settings, extract_images=False, continuous=False):
    '''
    This is the full pipeline which takes in an image and spits out the image with lane lines.
    If extract_images is called, it also returns a dictionary with the various images:
        'original': The original image
        'undis': The original image, undistorted
        'thresholded': The image with thresholds applied. This is black and white
        'warped_thresholded': The image thresholded image warped to a top-down view
        'thresholded_with_lines': warped_thresholded with the lines colored in and anything not a line no longer in the image
        'with_lines': The lines applied to the original image
    '''

    undis = undistort(image, camera_settings)

    thresholded_image = apply_threshold(undis)

    warped_thresholded_image = warp(thresholded_image)

    warped_thresholded_image_with_lines, lines, R = get_lines_from_thresholded_image(warped_thresholded_image, continuous=continuous)

    thresholded_image_with_lines = unwarp(warped_thresholded_image_with_lines)

    image_with_lines = apply_threshold_to_picture(thresholded_image_with_lines, image)

    image_with_lines_and_words = write_statistics(image_with_lines, lines, R)

    if extract_images:
        image_dict = {}
        image_dict['original'] = image
        image_dict['undis'] = undis
        image_dict['thresholded'] = thresholded_image
        image_dict['warped_thresholded'] = warped_thresholded_image
        image_dict['warped_thresholded_with_lines'] = warped_thresholded_image_with_lines
        image_dict['thresholded_with_lines'] = thresholded_image_with_lines
        image_dict['with_lines'] = image_with_lines
        image_dict['with_lines_and_words'] = image_with_lines_and_words

        return image_with_lines_and_words, image_dict

    return image_with_lines_and_words


def tile_images(image_array):
    '''
    Takes in an n x m array of pictures (n x m x w x h x 3) and stitches them together
    '''
    vertical_concat = np.concatenate(image_array, axis=1)
    horizontal_concat = np.concatenate(vertical_concat, axis=1)
    return horizontal_concat


if __name__ == '__main__':
    main()
