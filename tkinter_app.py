import os
# Force TensorFlow to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
MODEL_PATH = "unet_brain_tumor.keras"
IMG_SIZE = 256
UI_BG_COLOR = "#1e1e2e"
PANEL_BG_COLOR = "#313244"
TEXT_COLOR = "#cdd6f4"
BTN_COLOR_LOAD = "#89b4fa"
BTN_COLOR_PREDICT = "#a6e3a1"
BTN_COLOR_RESET = "#f38ba8"
FONT_MAIN = ("Segoe UI", 12, "bold")
FONT_TITLE = ("Segoe UI", 16, "bold")

class BrainTumorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Segmentation Interface")
        self.root.geometry("800x600")
        self.root.configure(bg=UI_BG_COLOR)
        
        self.model = None
        self.current_image_path = None
        self.raw_image = None
        
        # Tkinter image references to prevent garbage collection
        self.tk_original_img = None
        self.tk_mask_img = None
        
        self.setup_ui()
        self.load_keras_model()

    def load_keras_model(self):
        try:
            # compile=False allows us to load the model without defining custom losses/metrics
            self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            self.status_var.set("Status: Model Loaded Successfully (CPU Mode)")
        except Exception as e:
            self.status_var.set("Status: Error Loading Model!")
            messagebox.showerror("Model Error", f"Could not load '{MODEL_PATH}'.\nError: {e}")

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg=PANEL_BG_COLOR, pady=15)
        header.pack(fill=tk.X)
        title = tk.Label(header, text="🧠 Brain Tumor Segmentation", font=FONT_TITLE, bg=PANEL_BG_COLOR, fg=TEXT_COLOR)
        title.pack()

        # Image Display Area
        self.display_frame = tk.Frame(self.root, bg=UI_BG_COLOR, pady=20)
        self.display_frame.pack(expand=True, fill=tk.BOTH)

        # Original Image Panel
        self.orig_panel = tk.Frame(self.display_frame, bg=UI_BG_COLOR)
        self.orig_panel.pack(side=tk.LEFT, expand=True)
        tk.Label(self.orig_panel, text="Original Image", font=FONT_MAIN, bg=UI_BG_COLOR, fg=TEXT_COLOR).pack(pady=10)
        self.orig_label = tk.Label(self.orig_panel, bg=PANEL_BG_COLOR, width=35, height=15)
        self.orig_label.pack()

        # Predicted Mask Panel
        self.mask_panel = tk.Frame(self.display_frame, bg=UI_BG_COLOR)
        self.mask_panel.pack(side=tk.RIGHT, expand=True)
        tk.Label(self.mask_panel, text="Predicted Mask", font=FONT_MAIN, bg=UI_BG_COLOR, fg=TEXT_COLOR).pack(pady=10)
        self.mask_label = tk.Label(self.mask_panel, bg=PANEL_BG_COLOR, width=35, height=15)
        self.mask_label.pack()

        # Controls Area
        controls = tk.Frame(self.root, bg=PANEL_BG_COLOR, pady=20)
        controls.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_load = tk.Button(controls, text="📁 Load Image", font=FONT_MAIN, bg=BTN_COLOR_LOAD, fg="#11111b", 
                                  relief=tk.FLAT, width=15, cursor="hand2", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=30)

        self.btn_predict = tk.Button(controls, text="✨ Predict", font=FONT_MAIN, bg=BTN_COLOR_PREDICT, fg="#11111b", 
                                     relief=tk.FLAT, width=15, cursor="hand2", command=self.predict_mask)
        self.btn_predict.pack(side=tk.LEFT, padx=30)

        self.btn_reset = tk.Button(controls, text="🗑️ Reset", font=FONT_MAIN, bg=BTN_COLOR_RESET, fg="#11111b", 
                                   relief=tk.FLAT, width=15, cursor="hand2", command=self.reset_ui)
        self.btn_reset.pack(side=tk.RIGHT, padx=30)
        
        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Status: Waiting for model...")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bg="#181825", fg="#a6adc8", anchor="w", padx=10)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Brain MRI Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )
        if not file_path:
            return

        self.current_image_path = file_path
        self.raw_image = Image.open(self.current_image_path).convert("RGB")
        
        # Resize for UI Display
        display_img = self.raw_image.resize((256, 256), Image.Resampling.LANCZOS)
        self.tk_original_img = ImageTk.PhotoImage(display_img)
        
        self.orig_label.configure(image=self.tk_original_img, width=256, height=256)
        self.mask_label.configure(image='', width=35, height=15) # Clear previous mask
        self.status_var.set(f"Status: Loaded {os.path.basename(file_path)}")

    def predict_mask(self):
        if not self.current_image_path or self.raw_image is None:
            messagebox.showwarning("No Image", "Please load an image first!")
            return
            
        if self.model is None:
            messagebox.showerror("Model Missing", "Model is not loaded. Cannot predict.")
            return

        self.status_var.set("Status: Predicting...")
        self.root.update()

        try:
            # 1. Preprocess exactly as done in training
            img = tf.io.read_file(self.current_image_path)
            img = tf.image.decode_png(img, channels=3) # Or decode_jpeg depending on your test images
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            img = tf.cast(img, tf.float32) / 255.0
            
            # Expand dims to create a batch of 1
            img_expanded = tf.expand_dims(img, axis=0)

            # 2. Predict
            pred_mask = self.model.predict(img_expanded, verbose=0)[0]
            
            # 3. Binarize (threshold > 0.5)
            pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255

            # 4. Post-process for Tkinter Display
            # pred_mask_bin is (256, 256, 1), convert to (256, 256)
            mask_2d = np.squeeze(pred_mask_bin)
            
            mask_pil = Image.fromarray(mask_2d, mode='L')
            self.tk_mask_img = ImageTk.PhotoImage(mask_pil)
            
            self.mask_label.configure(image=self.tk_mask_img, width=256, height=256)
            self.status_var.set("Status: Prediction Complete.")

        except Exception as e:
            self.status_var.set("Status: Prediction Error!")
            messagebox.showerror("Prediction Error", str(e))

    def reset_ui(self):
        self.current_image_path = None
        self.raw_image = None
        self.tk_original_img = None
        self.tk_mask_img = None
        
        self.orig_label.configure(image='', width=35, height=15)
        self.mask_label.configure(image='', width=35, height=15)
        self.status_var.set("Status: Reset complete. Waiting for image...")


if __name__ == "__main__":
    root = tk.Tk()
    app = BrainTumorApp(root)
    root.mainloop()