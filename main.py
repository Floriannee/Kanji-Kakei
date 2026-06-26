import os
import time
import logging
import threading
import json
from datetime import datetime
import pandas as pd
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog, messagebox

# Config and Modules import
from config.settings import DB_FILE, OUTPUT_DIR
from database.db_manager import init_db, insert_receipt
from utils.image_processing import deskew_and_crop, opencv_to_pil
from inference.pipeline import ReceiptParser

# Initialize logger
logger = logging.getLogger("KanjiKakei.Main")

# Configure CustomTkinter UI Theme
ctk.set_appearance_mode("Dark")  # Options: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"

class KanjiKakeiApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure Window
        self.title("漢字家計 | Kanji-Kakei Receipt Reader")
        self.geometry("1100, 700")
        self.minsize(950, 600)
        
        # Initialize Core components
        self.parser = ReceiptParser()
        init_db()  # Ensure SQLite tables exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Variables
        self.uploaded_image_path = None
        self.processed_pil_image = None
        self.parsed_result = None
        self.start_time = None
        self.is_processing = False
        
        # Build UI Elements
        self.init_ui()
        logger.info("Application interface initialized successfully.")

    def init_ui(self):
        # Configure Grid Layout (1 row, 2 columns: Sidebar & Main Area)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # ----------------- SIDEBAR PANEL -----------------
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)
        
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Kanji-Kakei\n漢字家計", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.subtitle_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Student Receipt Scan & Advisor", 
            font=ctk.CTkFont(size=10, slant="italic"),
            text_color="gray"
        )
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 20))

        # Upload Button
        self.upload_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="📁 Upload Receipt", 
            command=self.upload_receipt,
            font=ctk.CTkFont(weight="bold")
        )
        self.upload_btn.grid(row=2, column=0, padx=20, pady=10)

        # Process Button
        self.process_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="⚙️ Process Receipt", 
            command=self.start_processing,
            state="disabled",
            font=ctk.CTkFont(weight="bold")
        )
        self.process_btn.grid(row=3, column=0, padx=20, pady=10)

        # Export CSV Button
        self.export_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="📥 Export to CSV", 
            command=self.export_csv,
            state="disabled",
            fg_color="green",
            hover_color="darkgreen",
            font=ctk.CTkFont(weight="bold")
        )
        self.export_btn.grid(row=4, column=0, padx=20, pady=10)
        
        # Real-time Stopwatch Label
        self.stopwatch_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Elapsed Time: 0.00s", 
            font=ctk.CTkFont(size=14, weight="bold", family="Courier New"),
            text_color="lightblue"
        )
        self.stopwatch_label.grid(row=6, column=0, padx=20, pady=(10, 20), sticky="s")

        # ----------------- MAIN VIEW AREA -----------------
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # Column Headers
        self.left_header = ctk.CTkLabel(
            self.main_frame, 
            text="Receipt Image Preview", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.left_header.grid(row=0, column=0, padx=10, pady=(0, 10))

        self.right_header = ctk.CTkLabel(
            self.main_frame, 
            text="Smart Read", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.right_header.grid(row=0, column=1, padx=10, pady=(0, 10))

        # Left Column: Image Canvas Frame
        self.image_frame = ctk.CTkFrame(self.main_frame)
        self.image_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        
        self.image_label = ctk.CTkLabel(
            self.image_frame, 
            text="No receipt uploaded yet.\nClick 'Upload Receipt' to start.", 
            text_color="gray"
        )
        self.image_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Right Column: Text Results display box
        self.result_textbox = ctk.CTkTextbox(
            self.main_frame, 
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.result_textbox.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.result_textbox.insert("1.0", "Inference results will be shown here.\n\nPlease upload a Japanese receipt image and click 'Process Receipt'.")

    def upload_receipt(self):
        """Open file dialog, run Classical OpenCV Deskew, and load image to preview pane."""
        file_path = filedialog.askopenfilename(
            title="Select Receipt Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not file_path:
            return
            
        self.uploaded_image_path = file_path
        logger.info(f"User uploaded image path: {file_path}")
        
        try:
            # 1. Run Classical Deskew Warp using OpenCV
            preprocessed_cv = deskew_and_crop(file_path)
            
            # 2. Convert to PIL for Tkinter rendering and pipeline usage
            self.processed_pil_image = opencv_to_pil(preprocessed_cv)
            
            # 3. Render in Preview Frame
            self.render_image_preview(self.processed_pil_image)
            
            # 4. Enable processing button
            self.process_btn.configure(state="normal")
            
            # Clear previous results
            self.result_textbox.delete("1.0", "end")
            self.result_textbox.insert("1.0", "Image preprocessed successfully!\nClick 'Process Receipt' to execute AI parsing.")
            
        except Exception as err:
            logger.error(f"Failed to load and preprocess receipt image: {err}")
            messagebox.showerror("Error", f"Failed to load receipt image: {err}")

    def render_image_preview(self, pil_img: Image.Image):
        """Scale and show the image in the GUI preview frame."""
        # Calculate aspect ratio scaling
        frame_width = self.image_frame.winfo_width()
        frame_height = self.image_frame.winfo_height()
        
        # Safety fallback if widget dimensions aren't initialized
        if frame_width < 100:
            frame_width = 400
        if frame_height < 100:
            frame_height = 500
            
        img_width, img_height = pil_img.size
        # Subtract padding to ensure the image fits perfectly inside the frame
        ratio = min((frame_width - 40) / img_width, (frame_height - 40) / img_height)
        
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # Convert to Tkinter PhotoImage using the original image.
        # CustomTkinter's CTkImage handles scaling natively based on Windows DPI scaling,
        # preventing the image from looking cropped or zoomed on high-DPI displays.
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_width, new_height))
        
        self.image_label.configure(image=ctk_img, text="")
        self.image_label.image = ctk_img  # Keep reference

    def start_processing(self):
        """Initiates background thread execution and starts the stopwatch."""
        if not self.processed_pil_image:
            messagebox.showwarning("Warning", "No receipt image is loaded.")
            return
            
        self.is_processing = True
        self.upload_btn.configure(state="disabled")
        self.process_btn.configure(state="disabled")
        self.export_btn.configure(state="disabled")
        
        self.result_textbox.delete("1.0", "end")
        self.result_textbox.insert("1.0", "Processing... Please wait.")
        
        # Start Stopwatch
        self.start_time = time.time()
        self.tick_stopwatch()
        
        # Execute multimodal API pipeline in background thread to prevent GUI lockup
        thread = threading.Thread(target=self.run_inference, args=(self.processed_pil_image,))
        thread.daemon = True
        thread.start()

    def tick_stopwatch(self):
        """Updates the elapsed time on the GUI every 50ms while processing is active."""
        if self.is_processing and self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.stopwatch_label.configure(text=f"Elapsed Time: {elapsed:.2f}s")
            self.after(50, self.tick_stopwatch)

    def run_inference(self, pil_img: Image.Image):
        """Runs the API parser client. Executes in the background thread."""
        try:
            result = self.parser.parse_receipt_image(pil_img)
            # Use after callback to safely update TKinter widgets from main thread
            self.after(0, self.on_inference_complete, result)
        except Exception as e:
            logger.error(f"Inference process failed: {e}")
            self.after(0, self.on_inference_failed, str(e))

    def on_inference_complete(self, result: dict):
        """Callback executing on the main GUI thread when inference finishes successfully."""
        self.is_processing = False
        elapsed = time.time() - self.start_time
        self.stopwatch_label.configure(text=f"Completed in: {elapsed:.2f}s")
        
        self.parsed_result = result
        
        # Update details textbox with formatted JSON
        self.result_textbox.delete("1.0", "end")
        formatted_json = json.dumps(result, indent=2, ensure_ascii=False)
        self.result_textbox.insert("1.0", formatted_json)
        
        # Insert record into SQLite database
        db_success = True
        try:
            insert_receipt(result, self.uploaded_image_path or "")
        except Exception as db_err:
            db_success = False
            logger.error(f"SQLite save failed: {db_err}")
            messagebox.showerror(
                "Database Error", 
                f"Failed to save receipt record to database.\n\nDetails: {db_err}"
            )
            
        # Re-enable inputs
        self.upload_btn.configure(state="normal")
        self.process_btn.configure(state="normal")
        self.export_btn.configure(state="normal")
        
        # Success prompt
        if db_success:
            messagebox.showinfo("Success", f"Processing complete in {elapsed:.2f} seconds. Data saved to database!")
        else:
            messagebox.showwarning("Warning", f"Processing complete in {elapsed:.2f} seconds, but database storage failed.")

    def on_inference_failed(self, error_message: str):
        """Callback executing on the main GUI thread if inference encounters an exception."""
        self.is_processing = False
        self.stopwatch_label.configure(text="Processing Error")
        
        self.result_textbox.delete("1.0", "end")
        self.result_textbox.insert("1.0", f"[ERROR] Inference process failed:\n\n{error_message}")
        
        # Re-enable inputs
        self.upload_btn.configure(state="normal")
        self.process_btn.configure(state="normal")
        
        messagebox.showerror(
            "API Inference Error", 
            f"The API parsing operation failed.\n\nPlease check your internet connection or verify your API key.\n\nDetails: {error_message}"
        )

    def export_csv(self):
        """Export current line items to a CSV file inside the outputs/ folder using pandas."""
        if not self.parsed_result or "items" not in self.parsed_result:
            messagebox.showwarning("Warning", "No parsed receipt data available to export.")
            return
            
        try:
            items_list = self.parsed_result["items"]
            
            # Build DataFrame
            df = pd.DataFrame(items_list)
            
            # Format filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"receipt_items_{timestamp}.csv"
            export_path = os.path.join(OUTPUT_DIR, filename)
            
            # Save using Pandas
            df.to_csv(export_path, index=False, encoding="utf-8-sig") # utf-8-sig for proper Japanese Excel rendering
            logger.info(f"Exported receipt data to CSV at path: {export_path}")
            
            messagebox.showinfo(
                "Export Successful", 
                f"Receipt items exported successfully!\n\nSaved to: {os.path.abspath(export_path)}"
            )
            
        except Exception as export_err:
            logger.error(f"Failed to export CSV: {export_err}")
            messagebox.showerror("Error", f"Failed to export CSV file: {export_err}")

if __name__ == "__main__":
    app = KanjiKakeiApp()
    app.mainloop()
