#from ultralytics import YOLO
from video_processing import VideoProcessing
from game_logic import GameLogic
from get_scores import GetScores

import cv2
import numpy as np
import os

from tkinter import *
from tkinter import font # submodule for dealing with fonts
from tkinter import ttk # submodule implementing a set of newer 'themed' widgets
from PIL import ImageTk, Image

class GUI:
    def __init__(self):
        self.display_size = 720
        self.video_processing = VideoProcessing()
        self.predict = GetScores()
        
        self.root = Tk()
        self.root.title("Dart Sense")
        self.root.columnconfigure(0, weight=1) # columns and rows are resized to fill window
        self.root.rowconfigure(0, weight=1)        

    def launch(self):
        self.menu_frame = ttk.Frame(self.root, padding="3 3 12 12")
        self.menu_frame.grid(column=0, row=0, sticky=(N, W, E, S))

        # add widgets for selecting game type
        ttk.Label(self.menu_frame, text="Game type:").grid(column=1, row=2, sticky=W)
        self.game_type = StringVar(self.menu_frame, "x01")
        game_type_combobox = ttk.Combobox(self.menu_frame, textvariable=self.game_type, values=["x01", "121", "cricket (not coded)"])
        game_type_combobox.grid(column=2, row=2, sticky=(W, E))
        
        ttk.Label(self.menu_frame, text="Legs (first to):").grid(column=1, row=3, sticky=W)
        self.legs = IntVar(self.menu_frame, 3)
        legs_entry = ttk.Entry(self.menu_frame, width=7, textvariable=self.legs)
        legs_entry.grid(column=2, row=3, sticky=(W, E))

        # if x01, add widget to allow user to enter starting score
        self.starting_score = IntVar(self.menu_frame, 501)
        ttk.Label(self.menu_frame, text="Starting score:").grid(column=1, row=4, sticky=W)
        starting_score_entry = ttk.Entry(self.menu_frame, width=7, textvariable=self.starting_score)
        starting_score_entry.grid(column=2, row=4, sticky=(W, E))

        # add widgets for selecting number of players
        ttk.Label(self.menu_frame, text="Players:").grid(column=1, row=1, sticky=W)
        self.num_players = IntVar(self.menu_frame, 1)
        num_players_entry = ttk.Entry(self.menu_frame, width=7, textvariable=self.num_players)
        num_players_entry.grid(column=2, row=1, sticky=(W, E))

        # add tickbox for calling out scores
        ttk.Label(self.menu_frame, text="Call scores:").grid(column=1, row=5, sticky=W)
        self.call_scores = BooleanVar(self.menu_frame, True)
        call_scores_checkbutton = ttk.Checkbutton(self.menu_frame, variable=self.call_scores)
        call_scores_checkbutton.grid(column=2, row=5, sticky=W)

        # create widget for ip webcam source
        ttk.Label(self.menu_frame, text="Webcam IP:").grid(column=1, row=6, sticky=W)
        self.ip = StringVar(self.menu_frame, "192.168.0.57:8080")
        ip_entry = ttk.Entry(self.menu_frame, width=15, textvariable=self.ip)
        ip_entry.grid(column=2, row=6, sticky=(W, E))

        # tick button for demo mode
        ttk.Label(self.menu_frame, text="Demo mode:").grid(column=1, row=7, sticky=W)
        self.demo_mode = BooleanVar(self.menu_frame, False)
        demo_mode_checkbutton = ttk.Checkbutton(self.menu_frame, variable=self.demo_mode)
        demo_mode_checkbutton.grid(column=2, row=7, sticky=W)


        def _start_game():
            self.scorer = GameLogic(self.game_type.get(), [player.get() for player in self.player_names], self.starting_score.get(), self.legs.get(), self.call_scores.get())
            self._game_screen()
            self.video_processing.start(self, self.ip.get(), self.scorer, np.array((1200, 1600)))

        # use number of players to create entry boxes for player names
        def update_player_names():
            self.menu_frame.destroy()
            self.player_frame = ttk.Frame(self.root, padding="3 3 12 12")
            self.player_frame.grid(column=0, row=0, sticky=(N, W, E, S))
            # add a back button which runs the launch method
            back_button = ttk.Button(self.player_frame, text="Back", command=self.launch)
            back_button.grid(column=1, row=1, sticky=W)
            ttk.Label(self.player_frame, text="Player names:").grid(column=1, row=2, sticky=W)
            self.player_names = []
            for i in range(1, self.num_players.get()+1):
                player_name = StringVar(self.player_frame, f'Player {i}')
                player_entry = ttk.Entry(self.player_frame, width=7, textvariable=player_name)
                player_entry.grid(column=2, row=i+1, sticky=(W, E))
                self.player_names.append(player_name)
            # destroy button
            add_player_names_button.destroy()
            # create button to start game
            start_game_button = ttk.Button(self.player_frame, text="Start", command=_start_game)
            start_game_button.grid(column=2, row=2+self.num_players.get(), sticky=W, columnspan=2)
            self.root.bind("<Return>", lambda e: _start_game())

        # create button to allow user to add player names
        add_player_names_button = ttk.Button(self.menu_frame, text="Next", command=update_player_names)
        add_player_names_button.grid(column=2, row=8, sticky=W, columnspan=2)
        self.root.bind("<Return>", lambda e: add_player_names_button.invoke())
        self.root.mainloop()
    

    def _game_screen(self):
        self.player_frame.destroy()

        self.mainframe = ttk.Frame(self.root, padding="3 3 12 12") # encompasses all parts of the UI
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        #self.mainframe.columnconfigure(0, weight=1)

        self.canvas = Canvas(self.mainframe, width=self.display_size, height=self.display_size)
        self.canvas.grid(column=1, row=self.scorer.num_players+5, columnspan=8, rowspan=8)

        self.heading_font = font.Font(family='Helvetica', size=12, underline=True)
        self.current_player_font = font.Font(family='Helvetica', size=12, weight='bold')
        self.text_font = font.Font(family='Helvetica', size=12)
        
        # change style for buttons to use Helvetica font
        s = ttk.Style()
        s.configure('TButton', font=('Helvetica', 9))
        s.configure('TCheckbutton', font=('Helvetica', 9))
        s.configure('TRadiobutton', font=('Helvetica', 9))
        s.configure('TLabel', font=('Helvetica', 9))

        if self.demo_mode.get():
            visibility

        # scorecard
        ttk.Label(self.mainframe, text=f'Race to {self.scorer.num_legs}', font=self.heading_font).grid(column=1, row=1, sticky=W)
        ttk.Label(self.mainframe, text="Legs", font=self.heading_font).grid(column=2, row=1, sticky=W)
        ttk.Label(self.mainframe, text="Scores", font=self.heading_font).grid(column=3, row=1, sticky=W)
        ttk.Label(self.mainframe, text="Avg", font=self.heading_font).grid(column=4, row=1, sticky=W)
        ttk.Label(self.mainframe, text="Visit", font=self.heading_font).grid(column=5, row=1, sticky=W, padx=(0, 120))

        # instantiate labels for each player, which will be updated later
        self.player_name_labels = []
        self.leg_labels = []
        self.score_labels = []
        self.avg_labels = []
        for i in range(self.scorer.num_players):
            name_label = ttk.Label(self.mainframe, text=self.scorer.player_names[i]) # assigning text as it won't change
            leg_label, score_label, avg_label = ttk.Label(self.mainframe), ttk.Label(self.mainframe), ttk.Label(self.mainframe)
            name_label.grid(column=1, row=i+2, sticky=W); leg_label.grid(column=2, row=i+2, sticky=W); score_label.grid(column=3, row=i+2, sticky=W); avg_label.grid(column=4, row=i+2, sticky=W)

            self.player_name_labels.append(name_label); self.leg_labels.append(leg_label); self.score_labels.append(score_label); self.avg_labels.append(avg_label)

        # current visit display
        self.dart_colours = ['cyan', 'yellow', 'deep pink']
        self.graphics_font = font.Font(family='Helvetica', size=10, weight='bold')

        self.visit_label = ttk.Label(self.mainframe, font=self.current_player_font)
        self.visit_label.grid(column=5, row=2, sticky=W)
        
        # radio buttons for display transformed, regular and live
        ttk.Label(self.mainframe, text="Display:", underline=0).grid(column=7, row=1, sticky=W)
        self.display_type = StringVar(self.mainframe, value='imageplane')
        original_button = ttk.Radiobutton(self.mainframe, text='Original', variable=self.display_type, value='imageplane')
        transform_button = ttk.Radiobutton(self.mainframe, text='Transformed', variable=self.display_type, value='boardplane')
        live_button = ttk.Radiobutton(self.mainframe, text='Live', variable=self.display_type, value='live')

        original_button.grid(column=7, row=2, sticky=W)
        transform_button.grid(column=7, row=3, sticky=W)
        live_button.grid(column=7, row=4, sticky=W)

        self.display_predictions = BooleanVar(self.mainframe, True)
        self.display_labels = BooleanVar(self.mainframe, True)
        
        self.fps_label = ttk.Label(self.mainframe)
        self.fps_label.grid(column=7, row=self.scorer.num_players+4, sticky=W)

        prediction_button = ttk.Checkbutton(self.mainframe, text="Predictions", variable=self.display_predictions, underline=0)
        prediction_button.grid(column=1, row=self.scorer.num_players+4, columnspan=2, sticky=W)

        labels_button = ttk.Checkbutton(self.mainframe, text="Labels", underline=0, variable=self.display_labels)
        labels_button.grid(column=2, row=self.scorer.num_players+4, columnspan=2, sticky=W)
        # add button for allowing user to add a dart that hasn't been detected
        add_button = ttk.Button(self.mainframe, text="Add dart", underline=0, command=lambda: self.video_processing.dart_coords_in_visit.append([0.5, 0.5]) if len(self.video_processing.dart_coords_in_visit) < 3 else None)
        add_button.grid(column=6, row=1, sticky=W)

        # add button for committing score, in event of bounce-outs
        commit_button = ttk.Button(self.mainframe, text="Commit score", underline=0, command=lambda: setattr(self.video_processing, 'wait_for_dart_removal', True))
        commit_button.grid(column=6, row=2, sticky=W)

        # button for saving image and labels in YOLO format
        save_button = ttk.Button(self.mainframe, text="Save data", underline=0, command=self.save_data)
        save_button.grid(column=6, row=3, sticky=W)

        # assign key bindings for buttons
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.root.bind('d', lambda e: transform_button.invoke() if self.display_type.get() == 'imageplane' else original_button.invoke() if self.display_type.get() == 'live' else live_button.invoke())
        self.root.bind('p', lambda e: prediction_button.invoke())
        self.root.bind('a', lambda e: add_button.invoke())
        self.root.bind('l', lambda e: labels_button.invoke())
        self.root.bind('s', lambda e: save_button.invoke())
        self.root.bind('<Return>', lambda e: commit_button.invoke())

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.root.bind("<Up>", self.on_arrow_key)
        self.root.bind("<Down>", self.on_arrow_key)
        self.root.bind("<Left>", self.on_arrow_key)
        self.root.bind("<Right>", self.on_arrow_key)
        self.root.bind("<Delete>", self.delete_object)

    def _get_segment_coords(self):
        # draw outline of a dart board to overlay on image
        radii = self.predict.scoring_radii*self.display_size
        angles = np.append(self.predict.segment_angles, 81)

        outer_distance = radii[-1] # double ring
        inner_distance = radii[2] # outer bull ring
        
        segment_coords = []

        for angle in angles:
            # a and o for adjacent and opposite sides of a right-angled triangle made by the segment
            outer_a = outer_distance*np.cos(np.deg2rad(angle)) 
            outer_o = (outer_distance**2 - outer_a**2)**0.5

            inner_a = inner_distance*np.cos(np.deg2rad(angle))
            inner_o = (inner_distance**2 - inner_a**2)**0.5

            if angle > 0:
                segment_coords.append(((self.display_size/2 + outer_a, self.display_size/2 + outer_o), (self.display_size/2 + inner_a, self.display_size/2 + inner_o)))
                segment_coords.append(((self.display_size/2 - outer_a, self.display_size/2 - outer_o), (self.display_size/2 - inner_a, self.display_size/2 - inner_o)))
            else:
                segment_coords.append(((self.display_size/2 - outer_a, self.display_size/2 + outer_o), (self.display_size/2 - inner_a, self.display_size/2 + inner_o)))
                segment_coords.append(((self.display_size/2 + outer_a, self.display_size/2 - outer_o), (self.display_size/2 + inner_a, self.display_size/2 - inner_o)))
        
        segment_coords = np.around(segment_coords)
        return segment_coords.astype(int)


    def _overlay_board_outline(self, image, display_size):
        colour = (255,0,0)
        # Define the coordinates of the board segments
        segment_coords = self._get_segment_coords()

        for segment in segment_coords:
            cv2.line(image, segment[0], segment[1], colour, 1)

        radii = np.around(self.predict.scoring_radii*display_size)
        radii = radii.astype(np.uint16)

        for radius in radii:
            cv2.circle(image, (int(display_size/2), int(display_size/2)), radius, colour, 1)
        
        return image
    
    def save_data(self):
        dir = 'data/images'

        # check files in dir to auto increment save file
        max_num = 0
        for file in os.listdir(dir):
            if file.startswith('dart_'):
                file_num = int(file[5:-4])
                max_num = max(max_num, file_num)

        new_num = str(max_num + 1)

        # Create the new file name
        image_path = os.path.join(dir, 'dart_'+ new_num + '.jpg')
        cv2.imwrite(image_path, self.img)
        
        label_path = os.path.join(dir.replace('images', 'labels'), 'dart_'+ new_num + '.txt')
        label = ""
        # transform from boardplane back to imageplane

        for dart in self.dart_coords_in_visit_imageplane:
            label += f'4 {np.round(dart[0], 6)} {np.round(dart[1], 6)} 0.025 0.025\n'
        
        for i in range(len(self.calibration_coords)):
            if np.all(self.calibration_coords[i] != -1):
                if i >= 4:
                    class_num = i+1
                else:
                    class_num = i
                label += f'{class_num} {np.round(self.calibration_coords[i][0], 6)} {np.round(self.calibration_coords[i][1], 6)} 0.025 0.025\n'
            
        with open(label_path, 'w') as f:
            f.write(label)
        
        print(f'Saved image to {image_path}')


    def move_object(self, x, y):
        if self.selected == 'calibration':
            self.canvas.coords(self.canvas_calibration[self.selected_index], x - 3, y - 3, x + 3, y + 3)
            self.video_processing.user_calibration[self.selected_index] = [x/self.display_size, y/self.display_size]

        else:
            self.canvas.coords(self.canvas_darts['ovals'][self.selected_index], x - 3, y - 3, x + 3, y + 3)
        
            if self.display_labels.get():
            # Update score text position and border
                self.canvas.coords(self.canvas_darts['texts'][self.selected_index], x-10, y+15)
                self.canvas.coords(self.canvas_darts['borders'][self.selected_index], self.canvas.bbox(self.canvas_darts['texts'][self.selected_index]))

            if self.display_type.get() == 'boardplane':
                self.video_processing.dart_coords_in_visit[self.selected_index] = [x/self.display_size, y/self.display_size]
            elif self.display_type.get() == 'imageplane':
                # as the coords are stored in the boardplane, need to convert them
                coords = np.array([[x/self.display_size, y/self.display_size]])
                transformed_coords = self.predict.transform_to_boardplane(self.H_matrix[0], coords, self.crop_size)
                self.video_processing.dart_coords_in_visit[self.selected_index] = [transformed_coords[0,0], transformed_coords[0,1]]


    def on_click(self, event):
        selected = self.canvas.find_overlapping(event.x-5, event.y-5, event.x+5, event.y+5)
        if selected:
            self.canvas.selected = selected[-1]  # select the top-most item
            if self.canvas.selected in self.canvas_darts['ovals']:
                self.selected = 'dart'
                self.selected_index = self.canvas_darts['ovals'].index(self.canvas.selected)
            elif self.canvas.selected in self.canvas_calibration:
                self.selected = 'calibration'
                self.selected_index = self.canvas_calibration.index(self.canvas.selected)
            else:
                self.selected_index = None
                self.selected = None


    def delete_object(self, event):
        if self.selected == 'calibration':
            self.video_processing.user_calibration[self.selected_index] = -np.ones(2)
            self.selected_index = None
        else:
            self.video_processing.dart_coords_in_visit.pop(self.selected_index)
            self.selected_index = None


    def on_arrow_key(self, event):
        if self.selected_index is not None:
            dx = 0
            dy = 0
            if event.keysym == 'Up':
                dy = -1
            elif event.keysym == 'Down':
                dy = 1
            elif event.keysym == 'Left':
                dx = -1
            elif event.keysym == 'Right':
                dx = 1
            
            if self.selected == 'calibration':
                x1, y1, x2, y2 = self.canvas.coords(self.canvas_calibration[self.selected_index])
            else:
                x1, y1, x2, y2 = self.canvas.coords(self.canvas_darts['ovals'][self.selected_index])
            
            x = ((x1 + x2) / 2) + dx
            y = ((y1 + y2) / 2) + dy
            self.move_object(x, y)
            

    def on_drag(self, event):
        if self.selected_index is not None:
            x, y = event.x, event.y
            self.move_object(x,y)


    def update_scoreboard(self, score, remaining, fps):
        darts = [dart for dart in self.video_processing.darts_in_visit if dart != '']
        if len(darts) == 0:
            self.visit_label.configure(text=f'')
        else:
            self.visit_label.configure(text=f'{", ".join(darts)} = {score}')
        self.visit_label.grid_configure(row=self.scorer.current_player+2)
        self.fps_label.configure(text=f'FPS: {str(fps)}')

        for i in range(self.scorer.num_players):
            if i == self.scorer.current_player:
                font_ = self.current_player_font
                score = remaining
            else:
                font_ = self.text_font
                score = self.scorer.scores[i]

            self.player_name_labels[i].configure(font=font_)
            self.leg_labels[i].configure(text=self.scorer.leg_scores[i], font=font_)
            self.score_labels[i].configure(text=score, font=font_)
            self.avg_labels[i].configure(text=int(np.round(self.scorer.averages[i])), font=font_)


    def _display_graphics(self, result, H_matrix, crop_start, crop_size, calibration_coords, dart_coords, score, remaining, fps):
        transformed_dart_coords = np.array(self.video_processing.dart_coords_in_visit)
        self.H_matrix = H_matrix
        self.crop_size = crop_size
        self.calibration_coords = calibration_coords
        # graphics to display no larger than 500 pixels
        self.img = result.orig_img[int(crop_start[1]):int(crop_start[1]+self.crop_size), int(crop_start[0]):int(crop_start[0]+self.crop_size)]

        calibration_mask = np.all(np.logical_and(self.calibration_coords>=0, self.calibration_coords<=1), axis=1) # remove any calibration points not detected or detected outside of square crop

        if self.display_type.get() == 'boardplane':
            GUI_img = cv2.warpPerspective(self.img, self.H_matrix[0], (self.crop_size, self.crop_size))
            GUI_img = cv2.resize(GUI_img, (self.display_size, self.display_size))
            GUI_img = self._overlay_board_outline(GUI_img, self.display_size)
            if self.display_predictions.get():
                dart_coords = transformed_dart_coords
                calibrations_to_plot = (np.round(self.predict.boardplane_calibration_coords[calibration_mask]*self.display_size)).astype(int)
        
        else:
            GUI_img = cv2.resize(self.img, (self.display_size, self.display_size))
            if self.display_predictions.get():
                calibrations_to_plot = self.calibration_coords[calibration_mask]
                calibrations_to_plot = (np.round(calibrations_to_plot*self.display_size)).astype(int) # convert to pixel coords
                if self.display_type.get() == 'imageplane' and len(transformed_dart_coords) > 0: # must convert the committed dart coordinates back from boardplane to imageplane
                    self.inv_matrix = np.linalg.inv(self.H_matrix[0])
                    self.dart_coords_in_visit_imageplane = self.predict.transform_to_boardplane(self.inv_matrix, transformed_dart_coords, self.crop_size)
                    dart_coords = self.dart_coords_in_visit_imageplane
        
        self.canvas.delete('all')
        GUI_img = ImageTk.PhotoImage(image=Image.fromarray(GUI_img[:,:,::-1]))
        self.canvas.create_image(0, 0, image=GUI_img, anchor=NW)

        if self.display_predictions.get():
            if dart_coords.shape != (0,):
                dart_coords = (np.round(dart_coords*self.display_size)).astype(int) # convert to pixel coords

            self.canvas_calibration = []
            for coords in calibrations_to_plot:
                x,y = coords
                oval = self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="", outline="white", width=1)
                self.canvas_calibration.append(oval)
            
            self.canvas_darts = {'ovals':[], 'texts':[], 'borders':[]}  # Store references to the ovals, texts and borders used to annotate the darts
            for i, coords in enumerate(dart_coords):
                x, y = coords
                oval = self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="", outline=self.dart_colours[i], width=2)
                self.canvas_darts['ovals'].append(oval)
            
                if self.display_type.get() != 'live' and self.display_labels.get():
                    #state = 'normal' if self.display_labels.get() else 'hidden'
 
                    # add text to canvas
                    score_text = self.canvas.create_text(x-10, y+15, text=self.video_processing.darts_in_visit[i], fill="black", font=self.graphics_font, anchor="w")
                    border = self.canvas.create_rectangle(self.canvas.bbox(score_text), fill=self.dart_colours[i])
                    self.canvas.tag_lower(score_text, self.canvas_darts['ovals'][0]) # keeps all ovals on top
                    self.canvas.tag_lower(border, score_text) # keeps text on top of border
                    self.canvas_darts['texts'].append(score_text)
                    self.canvas_darts['borders'].append(border)

            if self.display_type.get() == 'live':
                inference_text = self.canvas.create_text(5, self.display_size-15, text=f"Inference: {round(1000/result.speed['inference'], 1)} FPS", fill="black", font=self.graphics_font, anchor="w")
                border = self.canvas.create_rectangle(self.canvas.bbox(inference_text), fill="white")
                self.canvas.tag_lower(border, inference_text)

                if self.display_labels.get():
                    box_coords = np.array(result.boxes.xywh[:,0:2])
                    box_coords -= crop_start # adjust pixel coords for square crop
                    box_coords /= self.crop_size # convert back to normalised coords
                    box_coords = (np.round(box_coords*self.display_size)).astype(int)

                    dart_count = 0
                    for i, coords in enumerate(box_coords):
                        x,y = coords
                        # if dart, use dart colour, else use white
                        if result.boxes.cls[i] == 4:
                            colour = self.dart_colours[dart_count%3]
                            dart_count += 1
                        else:
                            colour = "white"
                        conf_text = self.canvas.create_text(x-10, y+15, text=str(round(float(result.boxes.conf[i]), 2)), fill="black", font=self.graphics_font, anchor="w")
                        border = self.canvas.create_rectangle(self.canvas.bbox(conf_text), fill=colour)
                        self.canvas.tag_lower(border, conf_text)

        self.update_scoreboard(score, remaining, fps)
        
        self.root.update()

if __name__ == '__main__':
    GUI().launch()