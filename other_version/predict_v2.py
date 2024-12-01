from predict_v1 import PredictV1
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import pandas as pd

from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line
import numpy as np
import cv2
from matplotlib import pyplot as plt
import itertools

class PredictV2():
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.segment_order_cartesian = np.array(([19, 1], [7, 18], [16, 4], [8, 13],
                                                 [6, 11], [10, 14], [15, 9], [2, 12], [17, 5]))
        
        # defining red and green colour ranges in HLS format (max values are 180,255,255)
        # hues for red shades are split at either end of the spectrum, hence the 2 ranges
        self.red_low = np.array([[0,26,71],
                                 [9,204,255]]) 
        self.red_high = np.array([[170,26,76],
                                  [180,178,255]])
        
        self.green = np.array([[40,26,38],
                               [80,178,255]])


    def process_yolo_output(self, output):
        dart_coords = []
        classes = output.boxes.cls
        boxes = output.boxes.xywh
        
        for i in range(len(classes)):
            if classes[i] == 4 and len(dart_coords) < 3:
                dart_coords.append([boxes[i][0], boxes[i][1]])
            
        return np.array(dart_coords)
    

    def find_intersection(self, r, theta):
        # find the intersection of 2 lines using Hough parameters
        y = (1 / np.sin(theta[0] - theta[1])) * (r[0] * np.cos(theta[1]) - r[1] * np.cos(theta[0]))
        x = (1 / np.cos(theta[0])) * (r[0] - y * np.sin(theta[0]))
        return [x,y]
    

    def delineate_segments(self, rgb_image, save_image=False):
        grey_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        edges = canny(grey_image, sigma=2, low_threshold=10, high_threshold=100, mode='nearest') # apply canny edge detection
        
        if save_image:
            fig, axes = plt.subplots(1, 3, figsize=(45, 15))
            ax = axes.ravel()
            plt.subplots_adjust(wspace=0, hspace=0.05)
            ax[0].imshow(edges, cmap='gray'); ax[1].imshow(rgb_image, cmap='gray')
            ax[0].set_axis_off(); ax[1].set_axis_off(); ax[2].set_axis_off()
            x_bounds = np.array((0, grey_image.shape[1])) # for plotting the lines on the image
            ax[1].set_xlim(x_bounds); ax[1].set_ylim((grey_image.shape[0], 0))
        else:
            ax=None

        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360) # Set a precision of 1/2 a degree
        hspace, theta, r = hough_line(edges, tested_angles)

        hough_distances, hough_angles, hough_points = [], [], [] # instantiate variables to store results of hough transform
        
        for _, angle, r in zip(*hough_line_peaks(hspace, theta, r, num_peaks=13, min_angle=24, min_distance=18)): # min_distance=18, min_angle=12 degrees
            hough_distances.append(r)
            hough_angles.append(angle)
            x = r*np.cos(angle)
            y = r*np.sin(angle)
            hough_points.append([x,y])
            if save_image:
                y0, y1 = (r - x_bounds * np.cos(angle)) / np.sin(angle)
                ax[1].plot(x_bounds, (y0, y1), '-r') # the ends of the lines on the image
        
        line_combinations = list(itertools.combinations(range(len(hough_angles)), 2))
        intersections = np.array([self.find_intersection((hough_distances[i[0]], hough_distances[i[1]]), (hough_angles[i[0]], hough_angles[i[1]])) for i in line_combinations])
        #intersections = np.array([intersection for intersection in intersections if np.sum(np.linalg.norm(intersection - intersections, axis=1) < 50) > 5])

        # filter out intersections that are not close to the other intersections, and then remove the offending lines from hough distances, angles and points
        line_counts_in_bad_intersections = np.zeros(len(hough_points))
        good_intersections = []
        for i in range(len(line_combinations)):
            intersection = intersections[i]
            line_combination = line_combinations[i]
            if np.sum(np.linalg.norm(intersection - intersections, axis=1) < 25) < 8:
                line_counts_in_bad_intersections[line_combination[0]] += 1
                line_counts_in_bad_intersections[line_combination[1]] += 1
            else:
                good_intersections.append(intersection)
        good_intersections = np.array(good_intersections)

        hough_distances = np.array([hough_distances[i] for i in range(len(hough_distances)) if line_counts_in_bad_intersections[i] < 8][:10])
        hough_angles = np.array([hough_angles[i] for i in range(len(hough_angles)) if line_counts_in_bad_intersections[i] < 8][:10])
        hough_points = np.array([hough_points[i] for i in range(len(hough_points)) if line_counts_in_bad_intersections[i] < 8][:10])

        bullseye_coords = np.mean(good_intersections, axis=0)
        cartesian_angles = np.array([np.rad2deg(np.arctan((point[1] - bullseye_coords[1]) / (point[0] - bullseye_coords[0]))) for point in hough_points])
        cartesian_angles.sort()

        if save_image:
            for i in range(len(hough_angles)):
                y0,y1 = (hough_distances[i] - x_bounds * np.cos(hough_angles[i])) / np.sin(hough_angles[i])
                ax[1].plot(x_bounds, (y0, y1), '-w') # white lines are in, red ones are out
            ax[1].plot(bullseye_coords[0], bullseye_coords[1], 'rx', markersize=10, markeredgewidth=3)

        return bullseye_coords, cartesian_angles, ax
    

    def determine_scoring_bed(self, number, bresenham_line, ax=None): 
        if number in [1,4,6,15,17,19,16,11,9,5]:
            segment_colour = 2 # corresponds to green - the dart is in a segment where doubles and trebles are green
            outcomes = {(0,1,0): {1:'DB', 2:'SB'}, (0,1,1):{2:'SB', 0:'S'}, (1,1,1):{0:'S', 2:'T'}, (1,1,2):{2:'T', 0:'S'},
                        (2,1,2):{0:'S', 2:'D'}, (2,1,3):{2:'D', 0:'miss'}, (3,1,3):{0:'miss'}}
        else:
            segment_colour = 1
            outcomes = {(0,1,0):{1:'DB', 2:'SB'}, (0,1,1):{2:'SB', 0:'S'}, (1,1,1):{0:'S', 1:'T'}, (1,2,1):{1:'T', 0:'S'},
                        (2,2,1):{0:'S', 1:'D'}, (2,3,1):{1:'D', 0:'miss'}, (3,3,1):{0:'miss'}}

        
        dart_pixel = bresenham_line[-1]
        
        colour_counts = np.zeros(3, np.uint8) # in order: neither red nor green, red, green

        current_colour = None
        current_colour_count = 0
        potential_colour = None
        potential_colour_count = 0

        for pixel in bresenham_line: # start from the bullseye, end at the dart location
            if pixel == current_colour:
                current_colour_count += 1
                if current_colour_count == 3:
                    colour_counts[current_colour] += 1
            
            else:
                if pixel == potential_colour:
                    potential_colour_count += 1
                    if potential_colour_count == 3:
                        current_colour = potential_colour
                        current_colour_count = potential_colour_count
                        colour_counts[potential_colour] += 1
                        potential_colour = None
                        potential_colour_count = 0
                else:
                    potential_colour = pixel
                    potential_colour_count = 1

        if segment_colour == 1 and np.all(colour_counts >= np.array((3,3,1))):
            return 'miss', colour_counts
        elif segment_colour == 2 and np.all(colour_counts >= np.array((3,1,3))):
            return 'miss', colour_counts
        
        try:
            bed = outcomes[tuple(colour_counts)][dart_pixel]
            return bed, colour_counts
        except KeyError:
            return -1, colour_counts

        
    def score(self, rgb_image, coords_for_darts, save_image=False, image_name=None):
        bullseye_coords, cartesian_angles, ax = self.delineate_segments(rgb_image, save_image=save_image)
                
        image_blurred = cv2.bilateralFilter(src=rgb_image, d=9, sigmaColor=75, sigmaSpace=75)
        hls_image = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2HLS)

        kernel = np.ones((3,3),np.uint8)

        low_red_mask = np.uint8(cv2.inRange(hls_image, self.red_low[0], self.red_low[1])/255)
        high_red_mask = np.uint8(cv2.inRange(hls_image, self.red_high[0], self.red_high[1])/255)
        red_mask = low_red_mask + high_red_mask
        red_mask = cv2.erode(red_mask, kernel, iterations = 2)
        red_mask = cv2.dilate(red_mask, kernel, iterations = 3)

        green_mask = np.uint8(cv2.inRange(hls_image, self.green[0], self.green[1])/255)
        green_mask = cv2.erode(green_mask, kernel, iterations = 2)
        green_mask = cv2.dilate(green_mask, kernel, iterations = 3)

        combined_mask = red_mask + green_mask*2
        combined_mask = np.where(combined_mask==3, 0, combined_mask) # pixels that are both red and green are set to neither

        if save_image:
            cmap = plt.cm.colors.ListedColormap(['black', 'red', 'green'])
            norm = plt.cm.colors.BoundaryNorm([0,1,2,3], cmap.N)
            ax[2].imshow(combined_mask, cmap=cmap, norm=norm) # show red green colour mask
 
        darts = []
        score = 0
        for dart_coords in coords_for_darts:
            dart_angle = np.rad2deg(np.arctan((dart_coords[1] - bullseye_coords[1]) / (dart_coords[0] - bullseye_coords[0])))
            if abs(dart_angle) >= cartesian_angles[-1]: # angle for 20 is a special case
                possible_numbers = np.array([3,20])
            else:
                try:
                    possible_numbers = self.segment_order_cartesian[max(np.where(cartesian_angles <= dart_angle)[0])]
                except ValueError: # segments not detected properly, so unable to score
                    darts.append('fail')
                    continue

            if all(possible_numbers == [6,11]):
                coord_index = 0
            else:
                coord_index = 1
            if dart_coords[coord_index] > bullseye_coords[coord_index]:
                number = possible_numbers[0]
            else:
                number = possible_numbers[1]
            
            rr,cc = line(int(bullseye_coords[1]), int(bullseye_coords[0]), int(dart_coords[1]), int(dart_coords[0]))
            bresenham_line = combined_mask[rr,cc]
            region, colour_counts = self.determine_scoring_bed(number, bresenham_line, ax)
            
            if region == -1:
                darts.append('fail')
                continue
            
            possible_score = {'DB':['DB',50], 'SB':['SB',25], 'S':['S'+str(number), number],
                              'T':['T'+str(number), number*3], 'D':['D'+str(number), number*2], 'miss':['miss',0]}
            if save_image:
                ax[1].plot(dart_coords[0], dart_coords[1], 'ro', markersize=8, markeredgewidth=2, markerfacecolor='none')
                ax[1].annotate(number, (dart_coords[0], dart_coords[1]),
                                color='black', bbox=dict(facecolor='white', edgecolor='white', alpha=0.7), fontsize=14, weight='bold', textcoords='offset points', xytext=(0,-20), ha='center')
                ax[2].plot(cc, rr, 'w-', linewidth=2)
                ax[2].annotate(possible_score[region][0], (dart_coords[0], dart_coords[1]),
                                color='black', bbox=dict(facecolor='white', edgecolor='white', alpha=0.7), fontsize=14, weight='bold', textcoords='offset points', xytext=(0,-20), ha='center')
                ax[2].annotate(colour_counts, (dart_coords[0], dart_coords[1]),
                                color='black', bbox=dict(facecolor='white', edgecolor='white', alpha=0.7), fontsize=14, weight='bold', textcoords='offset points', xytext=(0,-45), ha='center')
                
            darts.append(possible_score[region][0])
            score += possible_score[region][1]
        if save_image:
            plt.savefig(os.path.join(self.predict_dir, image_name), bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
        
        return darts, score


    def predict_scores(self, image_dir, conf, iou, model_results=None, save_images=False, compute_accuracy=True): # full workflow from model prediction to scoring to saving results
        # setting up save directories for results
        v1 = PredictV1(self.model_dir)
        ds_subset = os.path.split(image_dir)[1]
        ds = os.path.split(os.path.split(image_dir)[0])[1]
        weights = os.path.split(self.model_dir)[1].split(".")[0]
        model_num = os.path.split(os.path.split(os.path.split(self.model_dir)[0])[0])[1]
        self.predict_dir = 'prediction_examples/v2'
        
        # initialize score prediction dataframes
        predictions = pd.DataFrame(columns=['image_name', 'pred', 'gt'])

        totals = pd.Series({'dataset':ds,'subset':ds_subset, 'conf':conf, 'iou':iou, 'correct_visits':np.zeros(4).astype(int), 'total_visits':np.zeros(4).astype(int),
                            'correct_darts':0, 'darts_thrown':0, 'darts_predicted':0, 'missed_darts':0, 'extra_darts':0})

        if model_results is None:
            model = YOLO(self.model_dir)
            model_results = model(image_dir, stream=True, conf=conf, iou=iou)

        for output in model_results:
            pred_dart_coords = self.process_yolo_output(output)
            pred_darts, pred_score = self.score(output.orig_img[...,::-1], pred_dart_coords, save_images, os.path.basename(output.path)) # scores fully predicted using YOLO

            if compute_accuracy:
                gt_calibration_coords, gt_dart_coords = v1.get_coords_from_labels(output.path)
                gt_darts, gt_score, _, _ = v1.make_predictions(gt_calibration_coords, gt_dart_coords, output.orig_shape[::-1]) # scores inferred fully from GT labels
                # add results to dataframes
                predictions.loc[len(predictions.index)] = [os.path.basename(output.path), {'darts': pred_darts, 'score': pred_score}, {'darts': gt_darts, 'score': gt_score}]
                # update totals
                totals, _ = v1.update_totals(pred_darts, gt_darts, totals)
            
        return predictions, totals