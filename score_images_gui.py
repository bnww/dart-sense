"""
score_images_gui.py

Tkinter viewer for the Dart Sense pipeline over still images.

Same per-frame pipeline the live app uses, but broken out STEP BY STEP.
For each image the pipeline stages are rendered as individual frames you can
step through:

    1. Original image
    2. Centre square crop
    3. YOLO detections (darts + calibration points)
    4. Calibration points used for homography
    5. Board-plane warp (homography applied)
    6. Scored darts on board plane  -> "T20 S5 D16 = 97"

All settings are configured IN THE GUI (no command-line params):
    - Images folder (Browse) - searched recursively, incl. subfolders
    - Number of random images to sample (default 10)
    - YOLO confidence, weights file, display size, random seed

Change any setting, then press "Load / Re-run" to re-sample and re-process.
"Save settings" writes them to score_images_gui_settings.json and they are
reloaded automatically next time the app starts.

Navigate IMAGES with << / >> (Up/Down arrows).
Navigate STEPS  with < Step / Step > (Left/Right arrows).

Usage:
    python score_images_gui.py
"""

import os
import json
import random
import numpy as np
import cv2

from tkinter import *
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from ultralytics import YOLO
from get_scores import GetScores  # reuse the app's scoring logic unchanged


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'score_images_gui_settings.json')

DEFAULT_SETTINGS = {
    'images_dir': r'C:\Temp\Dart Data\cropped_images\800',
    'weights': 'weights.pt',
    'conf': 0.5,
    'display_size': 560,
    'num': 10,
    'seed': '',  # blank = random each run
}


def load_settings():
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_FILE, 'r') as f:
            settings.update(json.load(f))
    except (OSError, ValueError):
        pass
    return settings


class ScoreViewer:
    def __init__(self):
        self.settings = load_settings()

        # runtime state
        self.model = None
        self.predict = None
        self.loaded_weights = None
        self.files = []
        self.index = 0
        self.step = 0
        self.steps = []          # list of (title, bgr_image_or_None)
        self.visit_text = ""
        self.display_size = int(self.settings['display_size'])

        self._build_ui()
        self.run()               # initial load using saved/default settings
        self.root.mainloop()

    # ---------- UI ----------
    def _build_ui(self):
        self.root = Tk()
        self.root.title("Dart Sense - step-by-step image scorer")
        self.root.minsize(640, 600)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.frame = ttk.Frame(self.root, padding=10)
        self.frame.grid(row=0, column=0, sticky=(N, W, E, S))
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(3, weight=1)   # canvas row stretches

        # ----- settings panel -----
        s = ttk.LabelFrame(self.frame, text="Settings", padding=8)
        s.grid(row=0, column=0, sticky=(W, E), pady=(0, 8))

        self.var_dir = StringVar(value=self.settings['images_dir'])
        self.var_weights = StringVar(value=self.settings['weights'])
        self.var_conf = StringVar(value=str(self.settings['conf']))
        self.var_disp = StringVar(value=str(self.settings['display_size']))
        self.var_num = StringVar(value=str(self.settings['num']))
        self.var_seed = StringVar(value=str(self.settings['seed']))

        ttk.Label(s, text="Images folder:").grid(row=0, column=0, sticky=W, pady=2)
        ttk.Entry(s, textvariable=self.var_dir, width=46).grid(row=0, column=1, columnspan=3, sticky=(W, E), padx=4)
        ttk.Button(s, text="Browse...", command=self._browse_dir).grid(row=0, column=4, padx=4)

        ttk.Label(s, text="Weights:").grid(row=1, column=0, sticky=W, pady=2)
        ttk.Entry(s, textvariable=self.var_weights, width=46).grid(row=1, column=1, columnspan=3, sticky=(W, E), padx=4)
        ttk.Button(s, text="Browse...", command=self._browse_weights).grid(row=1, column=4, padx=4)

        ttk.Label(s, text="Num images:").grid(row=2, column=0, sticky=W, pady=2)
        ttk.Entry(s, textvariable=self.var_num, width=8).grid(row=2, column=1, sticky=W, padx=4)
        ttk.Label(s, text="Confidence:").grid(row=2, column=2, sticky=E, padx=4)
        ttk.Entry(s, textvariable=self.var_conf, width=8).grid(row=2, column=3, sticky=W, padx=4)

        ttk.Label(s, text="Render res:").grid(row=3, column=0, sticky=W, pady=2)
        ttk.Entry(s, textvariable=self.var_disp, width=8).grid(row=3, column=1, sticky=W, padx=4)
        ttk.Label(s, text="Seed (blank=random):").grid(row=3, column=2, sticky=E, padx=4)
        ttk.Entry(s, textvariable=self.var_seed, width=8).grid(row=3, column=3, sticky=W, padx=4)

        btns = ttk.Frame(s)
        btns.grid(row=4, column=0, columnspan=5, pady=(8, 0))
        ttk.Button(btns, text="Load / Re-run", command=self.run).grid(row=0, column=0, padx=6)
        ttk.Button(btns, text="Save settings", command=self.save_settings).grid(row=0, column=1, padx=6)

        # ----- viewer -----
        self.title_label = ttk.Label(self.frame, font=('Helvetica', 12, 'bold'))
        self.title_label.grid(row=1, column=0, pady=(0, 4))

        self.step_label = ttk.Label(self.frame, font=('Helvetica', 11))
        self.step_label.grid(row=2, column=0, pady=(0, 6))

        self.canvas = Canvas(self.frame, highlightthickness=1,
                             highlightbackground="#888", background="#222")
        self.canvas.grid(row=3, column=0, sticky=(N, W, E, S))
        self.canvas.bind('<Configure>', self._on_canvas_resize)

        self.score_label = ttk.Label(self.frame, font=('Helvetica', 14, 'bold'))
        self.score_label.grid(row=4, column=0, pady=10)

        step_nav = ttk.Frame(self.frame)
        step_nav.grid(row=5, column=0, pady=(0, 6))
        ttk.Button(step_nav, text="< Step", command=self.prev_step).grid(row=0, column=0, padx=6)
        self.step_counter = ttk.Label(step_nav, font=('Helvetica', 10))
        self.step_counter.grid(row=0, column=1, padx=12)
        ttk.Button(step_nav, text="Step >", command=self.next_step).grid(row=0, column=2, padx=6)
        self.var_grid = BooleanVar(value=False)
        ttk.Checkbutton(step_nav, text="Show all steps (2x3)",
                        variable=self.var_grid,
                        command=self.render).grid(row=0, column=3, padx=12)

        img_nav = ttk.Frame(self.frame)
        img_nav.grid(row=6, column=0)
        ttk.Button(img_nav, text="<< Prev image", command=self.prev_image).grid(row=0, column=0, padx=6)
        self.counter_label = ttk.Label(img_nav, font=('Helvetica', 10))
        self.counter_label.grid(row=0, column=1, padx=12)
        ttk.Button(img_nav, text="Next image >>", command=self.next_image).grid(row=0, column=2, padx=6)

        self.root.bind('<Left>', lambda e: self.prev_step())
        self.root.bind('<Right>', lambda e: self.next_step())
        self.root.bind('<Up>', lambda e: self.prev_image())
        self.root.bind('<Down>', lambda e: self.next_image())
        self.root.bind('<Escape>', lambda e: self.root.quit())

    def _browse_dir(self):
        d = filedialog.askdirectory(initialdir=self.var_dir.get() or '.')
        if d:
            self.var_dir.set(d)

    def _browse_weights(self):
        f = filedialog.askopenfilename(initialdir=os.path.dirname(self.var_weights.get()) or '.',
                                       filetypes=[('PyTorch weights', '*.pt'), ('All files', '*.*')])
        if f:
            self.var_weights.set(f)

    # ---------- settings ----------
    def _collect_settings(self):
        """Read GUI fields into self.settings, with validation."""
        try:
            self.settings = {
                'images_dir': self.var_dir.get(),
                'weights': self.var_weights.get(),
                'conf': float(self.var_conf.get()),
                'display_size': int(self.var_disp.get()),
                'num': int(self.var_num.get()),
                'seed': self.var_seed.get().strip(),
            }
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Num, confidence and display size must be numbers.")
            return False
        return True

    def save_settings(self):
        if not self._collect_settings():
            return
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(self.settings, f, indent=2)
            messagebox.showinfo("Saved", f"Settings saved to:\n{SETTINGS_FILE}")
        except OSError as e:
            messagebox.showerror("Save failed", str(e))

    # ---------- load / re-run ----------
    def run(self):
        if not self._collect_settings():
            return

        self.display_size = self.settings['display_size']

        # (re)load model only if weights changed
        weights = self.settings['weights']
        if self.model is None or self.loaded_weights != weights:
            try:
                self.model = YOLO(weights)
                self.predict = GetScores(weights)
                self.loaded_weights = weights
            except Exception as e:
                messagebox.showerror("Model load failed", str(e))
                return

        images_dir = self.settings['images_dir']
        all_files = []
        for root, _, names in os.walk(images_dir):
            for n in names:
                if n.endswith(IMG_EXTS):
                    all_files.append(os.path.join(root, n))
        if not all_files:
            messagebox.showerror("No images", f"No images found in:\n{images_dir}")
            return

        seed = self.settings['seed']
        if seed != '':
            try:
                random.seed(int(seed))
            except ValueError:
                random.seed(seed)
        else:
            random.seed()  # fresh random sample each run

        k = min(self.settings['num'], len(all_files))
        self.files = sorted(random.sample(all_files, k))
        self.index = 0
        self.load_current()

    # ---------- image navigation ----------
    def next_image(self):
        if self.files:
            self.index = (self.index + 1) % len(self.files)
            self.load_current()

    def prev_image(self):
        if self.files:
            self.index = (self.index - 1) % len(self.files)
            self.load_current()

    # ---------- step navigation ----------
    def next_step(self):
        if self.steps:
            self.step = (self.step + 1) % len(self.steps)
            self.render()

    def prev_step(self):
        if self.steps:
            self.step = (self.step - 1) % len(self.steps)
            self.render()

    # ---------- pipeline helpers ----------
    def _centre_square_crop(self, img):
        h, w = img.shape[:2]
        size = min(h, w)
        x0, y0 = (w - size) // 2, (h - size) // 2
        return img[y0:y0 + size, x0:x0 + size], (x0, y0), size

    def _draw_board_outline(self, img):
        ds = self.display_size
        radii = self.predict.scoring_radii * ds
        angles = np.append(self.predict.segment_angles, 81)
        outer, inner = radii[-1], radii[2]
        c = ds / 2
        for ang in angles:
            oa = outer * np.cos(np.deg2rad(ang)); oo = (outer**2 - oa**2) ** 0.5
            ia = inner * np.cos(np.deg2rad(ang)); io = (inner**2 - ia**2) ** 0.5
            if ang > 0:
                pts = [((c + oa, c + oo), (c + ia, c + io)),
                       ((c - oa, c - oo), (c - ia, c - io))]
            else:
                pts = [((c - oa, c + oo), (c - ia, c + io)),
                       ((c + oa, c - oo), (c + ia, c - io))]
            for p1, p2 in pts:
                cv2.line(img, tuple(np.round(p1).astype(int)),
                         tuple(np.round(p2).astype(int)), (255, 0, 0), 1)
        for r in np.round(radii).astype(int):
            cv2.circle(img, (int(c), int(c)), int(r), (255, 0, 0), 1)
        return img

    def _fit(self, bgr):
        return cv2.resize(bgr, (self.display_size, self.display_size))

    # ---------- build all pipeline steps ----------
    def _build_steps(self, path):
        ds = self.display_size
        steps = []
        visit_text = ""

        img = cv2.imread(path)
        if img is None:
            return [("could not read image", None)], ""

        # STEP 1: original
        steps.append(("1. Original image", self._fit(img)))

        # STEP 2: centre square crop
        crop, crop_start, crop_size = self._centre_square_crop(img)
        steps.append(("2. Centre square crop", self._fit(crop)))

        # YOLO inference
        result = self.model(crop, conf=self.settings['conf'], verbose=False)[0]
        calibration_coords, dart_coords = self.predict.process_yolo_output(result)

        # STEP 3: all YOLO detections
        det = self._fit(crop)
        for c in calibration_coords:
            if np.all(c >= 0):
                x, y = int(c[0] * ds), int(c[1] * ds)
                cv2.circle(det, (x, y), 5, (255, 255, 255), 2)
        for d in dart_coords:
            x, y = int(float(d[0]) * ds), int(float(d[1]) * ds)
            cv2.circle(det, (x, y), 6, (0, 255, 255), 2)
        steps.append(("3. YOLO detections (white=calib, yellow=darts)", det))

        detected = np.count_nonzero(np.all(calibration_coords >= 0, axis=1))

        # STEP 4: calibration points only
        calib = self._fit(crop)
        for c in calibration_coords:
            if np.all(c >= 0):
                x, y = int(c[0] * ds), int(c[1] * ds)
                cv2.circle(calib, (x, y), 6, (0, 255, 0), 2)
        steps.append((f"4. Calibration points ({detected}/4 found)", calib))

        if detected < 4:
            visit_text = f"only {detected}/4 calibration points - cannot transform"
            return steps, visit_text

        # homography + transform
        H_matrix = self.predict.find_homography(calibration_coords, crop_size)
        board_coords = self.predict.transform_to_boardplane(H_matrix[0], dart_coords, crop_size)
        darts, score = self.predict.score(np.array(board_coords))

        # STEP 5: board-plane warp + outline
        warped = cv2.warpPerspective(crop, H_matrix[0], (crop_size, crop_size))
        warped = self._fit(warped)
        warped_outline = self._draw_board_outline(warped.copy())
        steps.append(("5. Homography warp to board plane", warped_outline))

        # STEP 6: scored darts on board plane
        scored = self._draw_board_outline(warped.copy())
        for (dx, dy), label in zip(board_coords, darts):
            px, py = int(round(dx * ds)), int(round(dy * ds))
            cv2.circle(scored, (px, py), 6, (0, 255, 255), 2)
            cv2.putText(scored, label, (px - 12, py + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        steps.append(("6. Scored darts", scored))

        visit = ' '.join(darts) if darts else '(no darts detected)'
        visit_text = f"{visit}  =  {score}"
        return steps, visit_text

    def _to_photo(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    # ---------- render ----------
    def _on_canvas_resize(self, event):
        # re-render at the new canvas size (skip tiny spurious events)
        if self.steps and (event.width > 10 and event.height > 10):
            self.render()

    def _canvas_dims(self):
        w = max(self.canvas.winfo_width(), 1)
        h = max(self.canvas.winfo_height(), 1)
        return w, h

    def load_current(self):
        path = self.files[self.index]
        self.steps, self.visit_text = self._build_steps(path)
        self.step = 0
        self.render()

    def render(self):
        if not self.steps:
            return
        fname = os.path.relpath(self.files[self.index], self.settings['images_dir'])
        self.title_label.configure(text=fname)
        self.counter_label.configure(text=f"image {self.index + 1} / {len(self.files)}")
        self.score_label.configure(text=self.visit_text)
        self.canvas.delete('all')

        if self.var_grid.get():
            self._render_grid()
        else:
            self._render_single()

    def _fit_to(self, bgr, box_w, box_h):
        """Scale BGR to fit within box (preserve aspect). Returns (photo, w, h)."""
        h, w = bgr.shape[:2]
        scale = min(box_w / w, box_h / h)
        nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
        resized = cv2.resize(bgr, (nw, nh))
        return self._to_photo(resized), nw, nh

    def _render_single(self):
        title, bgr = self.steps[self.step]
        self.step_label.configure(text=title)
        self.step_counter.configure(text=f"step {self.step + 1} / {len(self.steps)}")
        cw, ch = self._canvas_dims()
        if bgr is not None:
            self._img, iw, ih = self._fit_to(bgr, cw, ch)  # keep ref
            self.canvas.create_image(cw / 2, ch / 2, image=self._img, anchor=CENTER)
        else:
            self.canvas.create_text(cw / 2, ch / 2, text="no image", fill="#ccc")

    def _render_grid(self):
        """All pipeline steps as thumbnails in a 2-row x 3-col grid, filling the canvas."""
        self.step_label.configure(text="All steps")
        self.step_counter.configure(text=f"{len(self.steps)} steps")

        cols, rows = 3, 2
        pad = 8
        cw, ch = self._canvas_dims()
        cell_w = (cw - pad * (cols + 1)) // cols
        cell_h = (ch - pad * (rows + 1)) // rows

        self._thumbs = []  # keep refs
        for i, (title, bgr) in enumerate(self.steps[:cols * rows]):
            r, c = divmod(i, cols)
            cx = pad + c * (cell_w + pad) + cell_w / 2
            cy = pad + r * (cell_h + pad) + (cell_h - 16) / 2
            if bgr is not None:
                photo, iw, ih = self._fit_to(bgr, cell_w, cell_h - 16)
                self._thumbs.append(photo)
                self.canvas.create_image(cx, cy, image=photo, anchor=CENTER)
            self.canvas.create_text(cx, pad + r * (cell_h + pad) + cell_h - 4,
                                    text=title, fill="#ccc", anchor=S,
                                    font=('Helvetica', 8))


def main():
    ScoreViewer()


if __name__ == '__main__':
    main()