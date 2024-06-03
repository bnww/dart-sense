import cv2
import numpy as np

class GetScores():
    def __init__(self, model_dir="runs\\train180\\weights\\best.pt"):
        self.model_dir = model_dir

        self.class_names = {0: '20', 1: '3', 2: '11', 3: '6', 4: 'dart'}
        
        # dart board measurements in mm
        ring = 10.0 # width of the double and treble rings
        bullseye_wire = 1.6 # width of the bullseye wires
        wire = 1.0 # width of other wires

        self.scoring_names = np.array(['DB', 'SB', 'S', 'T', 'S', 'D', 'miss'])
        self.scoring_radii = np.array([0, 6.35, 15.9, 107.4-ring, 107.4, 170.0-ring, 170.0]) # inside radius of the corresponding regions in scoring_names
        self.scoring_radii[1:3] += (bullseye_wire/2) # add on half the width of the bullseye wire
        #self.scoring_radii[3:] += (wire/2) # found that it's more accurate to leave off the wire width for the double and treble regions

        self.scoring_radii /= 451.0 # divide by the diameter of the dart board to normalize between 0-1

        self.segment_angles = np.array([-9, 9, 27, 45, 63, -81, -63, -45, -27]) # minimum angle for corresponding pairs of numbered segments below
        self.segment_numbers = np.array(([6,11], [10,14], [15,9], [2,12], [17,5], [19,1], [7,18], [16,4], [8,13]))     

        # computing the boardplane calibration coordinates using cosx = a/h
        self.boardplane_calibration_coords = -np.ones((6, 2))
        h = self.scoring_radii[-1]

        # for 20 & 3
        a = h*np.cos(np.deg2rad(81))
        o = (h**2 - a**2)**0.5
        self.boardplane_calibration_coords[0] = [0.5 - a, 0.5 - o]
        self.boardplane_calibration_coords[1] = [0.5 + a, 0.5 + o]

        # for 11 & 6
        a = h*np.cos(np.deg2rad(-9))
        o = (h**2 - a**2)**0.5
        self.boardplane_calibration_coords[2] = [0.5 - a, 0.5 + o]
        self.boardplane_calibration_coords[3] = [0.5 + a, 0.5 - o]

        # for 9 & 15
        a = h*np.cos(np.deg2rad(27))
        o = (h**2 - a**2)**0.5
        self.boardplane_calibration_coords[4] = [0.5 - a, 0.5 - o]
        self.boardplane_calibration_coords[5] = [0.5 + a, 0.5 + o]
    

    def __str__(self):
        return f'Model directory: {self.model_dir}\n\
            Boardplane calibration coordinates: {self.boardplane_calibration_coords}\n\
            Scoring radii: {self.scoring_radii}\n\
            Segment angles: {self.segment_angles}'
                

    def process_yolo_output(self, output):
        calibration_coords = -np.ones((6, 2))
        dart_coords = []
        classes = output.boxes.cls
        boxes = output.boxes.xywhn
        conf = output.boxes.conf
        
        for i in range(len(classes)):
            if classes[i] == 4 and len(dart_coords) < 3:
                dart_coords.append([boxes[i][0], boxes[i][1]])
            elif classes[i] == 4:
                continue
            else:
                if conf[i] < 0.85:
                    continue
                calibration_i = int(classes[i].item())
                if calibration_i > 4:
                    calibration_i -= 1

                if np.all(calibration_coords[calibration_i] == -1): # don't overwrite if respective calibration point already detected
                    calibration_coords[calibration_i] = boxes[i][:2]
        
        dart_coords = np.array(dart_coords)

        return calibration_coords, dart_coords
    

    def find_homography(self, calibration_coords, image_shape):
        mask = np.all(np.logical_and(calibration_coords >= 0, calibration_coords <= 1), axis=1)
        H_matrix = cv2.findHomography(calibration_coords[mask]*image_shape, self.boardplane_calibration_coords[mask]*image_shape)
        return H_matrix


    def transform_to_boardplane(self, matrix, dart_coords, image_shape):
        # convert to actual image coordinates, so that the homography matrix can be used to transform the image for saving results
        if len(dart_coords)==0:
            return dart_coords
       
        homogenous_coords = np.concatenate((dart_coords*image_shape, np.ones((dart_coords.shape[0], 1))), axis=1).T
        transformed_darts = matrix @ homogenous_coords
        transformed_darts /= transformed_darts[-1] # divide by w to get x and y coords in 2D space
        transformed_darts = transformed_darts[:-1].T
        transformed_darts /= image_shape # normalize again for scoring function

        return transformed_darts


    def score(self, transformed_darts):
        # initialize variables
        darts = ['' for _ in range(len(transformed_darts))]
        score = 0

        if len(darts) == 0:
            return darts, score

        mask = transformed_darts[:,0] == 0.5
        transformed_darts[mask,0] += 0.00001 # prevent division by zero error
        
        # find the angles of the dart locations relative to the center
        angles = np.rad2deg(np.arctan((transformed_darts[:,1]-0.5)/(transformed_darts[:,0]-0.5)))
        angles = np.where(angles > 0, np.floor(angles), np.ceil(angles))

        for i in range(len(transformed_darts)):
            dart_coords = transformed_darts[i]
            
            # use computed angle to work out which numbered segment the dart lies in
            if abs(angles[i]) >= 81:
                possible_numbers = np.array([3,20])
            else:
                possible_numbers = self.segment_numbers[np.where(self.segment_angles == max(self.segment_angles[self.segment_angles <= angles[i]]))][0]

            # angle can only narrow down to 2 possible numbered segments, so use coordinate values to determine which of the 2 it is
            if all(possible_numbers == [6,11]):
                coord_index = 0
            else:
                coord_index = 1
            if dart_coords[coord_index] > 0.5:
                number = possible_numbers[0]
            else:
                number = possible_numbers[1]
            
            # compute euclidean distance from the center, and thus the scoring region e.g single, double, treble, bullseye, miss
            distance = ((dart_coords[0]-0.5)**2 + (dart_coords[1]-0.5)**2)**0.5
            region = self.scoring_names[np.argmax(self.scoring_radii[distance > self.scoring_radii])]

            scores = {'DB':['DB',50], 'SB':['SB',25], 'S':['S'+str(number), number],
                      'T':['T'+str(number), number*3], 'D':['D'+str(number), number*2], 'miss':['miss',0]}
            
            darts[i] = scores[region][0]
            score += scores[region][1]
                
        return darts, score