import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

class BrainSegApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Segmentation Interface")
        self.root.geometry("900x700")
        self.root.configure(bg="#2E3440") # Dark theme background
        
        # Color Palette
        self.colors = {
            "bg": "#2E3440",
            "panel": "#3B4252",
            "text": "#ECEFF4",
            "border": "#4C566A",
            "load_btn": "#88C0D0",
            "predict_btn": "#A3BE8C",
            "reset_btn": "#BF616A",
            "btn_text": "#2E3440"
        }
        
        self.image_path = None
        self.cv_image = None
        self.display_photo = None
        
        # Initialize YOLO model
        try:
            self.model = YOLO("brainseg26n-seg.pt")
        except Exception as e:
            self.model = None
            print(f"Warning: Ensure 'brainseg26n-seg.pt' is in the same directory. Error: {e}")

        self.setup_ui()

    def setup_ui(self):
        # --- Title Frame ---
        title_frame = tk.Frame(self.root, bg=self.colors["bg"])
        title_frame.pack(side=tk.TOP, fill=tk.X, pady=20)
        
        title_label = tk.Label(
            title_frame, 
            text="YOLO Brain Tumor Segmentation", 
            font=("Helvetica", 20, "bold"),
            bg=self.colors["bg"], 
            fg=self.colors["text"]
        )
        title_label.pack()

        # --- Image Display Frame ---
        self.image_frame = tk.Frame(
            self.root, 
            bg=self.colors["panel"], 
            bd=4, 
            relief=tk.FLAT,
            highlightbackground=self.colors["border"],
            highlightthickness=2
        )
        self.image_frame.pack(pady=10, padx=40, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(
            self.image_frame, 
            text="No Image Loaded", 
            font=("Helvetica", 14), 
            bg=self.colors["panel"], 
            fg=self.colors["text"]
        )
        self.image_label.pack(expand=True)

        # --- Buttons Frame ---
        btn_frame = tk.Frame(self.root, bg=self.colors["bg"])
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=30)

        # Container to center buttons
        center_btn_frame = tk.Frame(btn_frame, bg=self.colors["bg"])
        center_btn_frame.pack(anchor=tk.CENTER)

        # Load Button
        self.load_btn = tk.Button(
            center_btn_frame, 
            text="Load Image", 
            font=("Helvetica", 12, "bold"),
            bg=self.colors["load_btn"], 
            fg=self.colors["btn_text"], 
            activebackground="#81A1C1",
            relief=tk.FLAT,
            padx=20, pady=10,
            command=self.load_image
        )
        self.load_btn.grid(row=0, column=0, padx=15)

        # Predict Button
        self.predict_btn = tk.Button(
            center_btn_frame, 
            text="Predict", 
            font=("Helvetica", 12, "bold"),
            bg=self.colors["predict_btn"], 
            fg=self.colors["btn_text"],
            activebackground="#8FBCBB",
            relief=tk.FLAT,
            padx=20, pady=10,
            command=self.predict_image
        )
        self.predict_btn.grid(row=0, column=1, padx=15)

        # Reset Button
        self.reset_btn = tk.Button(
            center_btn_frame, 
            text="Reset", 
            font=("Helvetica", 12, "bold"),
            bg=self.colors["reset_btn"], 
            fg=self.colors["text"], 
            activebackground="#D08770",
            relief=tk.FLAT,
            padx=20, pady=10,
            command=self.reset_ui
        )
        self.reset_btn.grid(row=0, column=2, padx=15)

    def load_image(self):
        filepath = filedialog.askopenfilename(
            title="Select an Image", 
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not filepath:
            return
            
        self.image_path = filepath
        self.cv_image = cv2.imread(self.image_path)
        self.show_image(self.cv_image)

    def predict_image(self):
        if self.cv_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
            
        if self.model is None:
            messagebox.showerror("Error", "YOLO model not loaded properly. Ensure 'brainseg26n-seg.pt' exists.")
            return

        try:
            # Run inference using the Ultralytics YOLO API
            results = self.model.predict(self.image_path, conf=0.25, verbose=False)
            
            # res[0].plot() returns a BGR numpy array with drawn segmentation masks
            annotated_img_bgr = results[0].plot()
            self.show_image(annotated_img_bgr)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{e}")

    def reset_ui(self):
        self.image_path = None
        self.cv_image = None
        self.image_label.config(image='', text="No Image Loaded")
        self.display_photo = None

    def show_image(self, img_array):
        # Convert BGR to RGB for Tkinter/Pillow
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Calculate dynamic resizing to fit the frame while maintaining aspect ratio
        frame_width = self.image_frame.winfo_width() - 20
        frame_height = self.image_frame.winfo_height() - 20
        
        # Fallback if window hasn't drawn properly yet
        if frame_width < 100 or frame_height < 100:
            frame_width, frame_height = 800, 500

        pil_img.thumbnail((frame_width, frame_height), Image.Resampling.LANCZOS)
        
        self.display_photo = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=self.display_photo, text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = BrainSegApp(root)
    root.mainloop()