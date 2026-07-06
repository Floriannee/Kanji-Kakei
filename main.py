import os
import time
import logging
import threading
import webbrowser  # To open the interactive HTML Dashboard
from datetime import datetime
from collections import defaultdict
import http.server
import socketserver
import json
import base64

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image

# Config and Modules import
from config.settings import OUTPUT_DIR
from database.db_manager import init_db, insert_receipt
from database.csv_manager import init_csv, append_items_to_csv, load_all_items
from utils.image_processing import deskew_and_crop, opencv_to_pil
from inference.pipeline import ReceiptParser

# Initialize logger
logger = logging.getLogger("KanjiKakei.Main")

# Configure CustomTkinter UI Theme
ctk.set_appearance_mode("Dark")  # Options: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"

# Preferred display order for item categories. Anything outside this list is
# sorted alphabetically and appended at the end.
CATEGORY_ORDER = [
    "Groceries", "Drink", "Snack", "Dining Out", "Daily Essentials", 
    "Clothes", "Personal Care", "Stationery", "Leisure", "Souvenirs", 
    "Tax", "Other"
]


def sorted_categories(categories) -> list:
    """Return the given category names ordered using CATEGORY_ORDER, with any
    unrecognized categories sorted alphabetically at the end."""
    known = [c for c in CATEGORY_ORDER if c in categories]
    unknown = sorted(c for c in categories if c not in CATEGORY_ORDER)
    return known + unknown


def safe_int(value) -> int:
    """Best-effort conversion of a CSV/JSON price field to an int for display/summing."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


class ReceiptApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window settings
        self.title("Kanji-Kakei (漢字家計)")
        self.geometry("1100x680")
        self.minsize(950, 600)

        # Initialize global layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=4)
        self.grid_rowconfigure(0, weight=1)

        # Non-UI internal properties
        self.current_image_path = None
        self.processing_thread = None

        # Create sub-panels
        self.setup_sidebar()
        self.setup_main_content()

        # Database and file system initialization
        init_db()
        init_csv()
        self.refresh_history_table()

    def setup_sidebar(self):
        """Create the left vertical navigation panel with controls."""
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # App title
        self.logo_label = ctk.CTkLabel(
            self.sidebar, text="Kanji-Kakei\n漢字家計", font=ctk.CTkFont(size=22, weight="bold")
        )
        self.logo_label.pack(padx=20, pady=(30, 40))

        # Core functionality buttons
        self.btn_browse = ctk.CTkButton(
            self.sidebar, text="📁 Select Receipt Image", height=40, command=self.browse_image
        )
        self.btn_browse.pack(padx=20, pady=10, fill="x")

        self.btn_scan = ctk.CTkButton(
            self.sidebar,
            text="✨ Run OCR & Analysis",
            height=40,
            state="disabled",
            fg_color="#27ae60",
            hover_color="#219653",
            command=self.start_processing_thread,
        )
        self.btn_scan.pack(padx=20, pady=10, fill="x")

        # This button launches the interactive HTML Dashboard
        self.btn_export = ctk.CTkButton(
            self.sidebar,
            text="📊 Open Interactive Dashboard",
            height=40,
            fg_color="#2980b9",
            hover_color="#1f6696",
            command=self.open_html_dashboard,
        )
        self.btn_export.pack(padx=20, pady=(10, 40), fill="x")

        # App status footer
        self.status_label = ctk.CTkLabel(
            self.sidebar, text="Ready", font=ctk.CTkFont(size=12), text_color="#aaaaaa"
        )
        self.status_label.pack(side="bottom", padx=20, pady=20)

    def setup_main_content(self):
        """Create the right spacious area containing tabs for image view and global history."""
        self.main_notebook = ctk.CTkTabview(self)
        self.main_notebook.grid(row=0, column=1, sticky="nsew", padx=20, pady=15)

        # Define individual tabs
        self.tab_preview = self.main_notebook.add("Image Preview")
        self.tab_history = self.main_notebook.add("All Expenses History")

        # Layout for Tab 1: Image Preview
        self.tab_preview.grid_rowconfigure(0, weight=1)
        self.tab_preview.grid_columnconfigure(0, weight=1)

        self.preview_placeholder = ctk.CTkLabel(
            self.tab_preview,
            text="No receipt selected.\nClick 'Select Receipt Image' to load an image file.",
            font=ctk.CTkFont(size=14, slant="italic"),
            text_color="#888888",
        )
        self.preview_placeholder.grid(row=0, column=0, sticky="nsew")

        # Layout for Tab 2: Expense History
        self.tab_history.grid_rowconfigure(1, weight=1)
        self.tab_history.grid_columnconfigure(0, weight=1)

        # Context summary description
        self.history_title = ctk.CTkLabel(
            self.tab_history,
            text="All historical scanned records matching the global CSV output dataset:",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        )
        self.history_title.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        # Main multi-column list box wrapper
        self.history_textbox = ctk.CTkTextbox(self.tab_history, font=ctk.CTkFont(family="Courier", size=12))
        self.history_textbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    def browse_image(self):
        """Open file dialog window allowing user to locate and pick supported local image types."""
        file_types = [("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All Files", "*.*")]
        selected_file = filedialog.askopenfilename(title="Select Receipt File", filetypes=file_types)

        if not selected_file:
            return

        self.current_image_path = selected_file
        logger.info(f"User picked file: {self.current_image_path}")

        # Update controls and show selected item inside UI window
        filename = os.path.basename(self.current_image_path)
        self.status_label.configure(text=f"Loaded: {filename}", text_color="#3498db")
        self.btn_scan.configure(state="normal")
        self.render_image_preview()

    def render_image_preview(self):
        """Load and adjust image to scale properly without distortions inside the widget frame."""
        for widget in self.tab_preview.winfo_children():
            widget.destroy()

        try:
            pil_img = Image.open(self.current_image_path)
            # Fetch canvas scale bound constraints
            canvas_w = self.main_notebook.winfo_width() - 60
            canvas_h = self.sidebar.winfo_height() - 120

            # Provide reasonable safe standard defaults if main engine loop hasn't fully rendered constraints yet
            if canvas_w < 200:
                canvas_w = 600
            if canvas_h < 200:
                canvas_h = 500

            img_w, img_h = pil_img.size
            scale_ratio = min(canvas_w / img_w, canvas_h / img_h)
            scale_ratio = max(scale_ratio, 0.02)  # Avoid zero DivisionError edgecases

            display_w = int(img_w * scale_ratio)
            display_h = int(img_h * scale_ratio)

            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(display_w, display_h))

            lbl_container = ctk.CTkLabel(self.tab_preview, image=ctk_img, text="")
            lbl_container.image = ctk_img
            lbl_container.pack(expand=True, fill="both", padx=10, pady=10)

            self.main_notebook.set("Image Preview")

        except Exception as err:
            logger.error(f"Failed rendering uploaded asset path preview frame: {err}", exc_info=True)
            messagebox.showerror("Preview Failure", f"An error occurred loading image display preview:\n{err}")

    def start_processing_thread(self):
        """Initialize separate system background thread execution routine preventing UI freezes."""
        if not self.current_image_path:
            return

        self.btn_scan.configure(state="disabled", text="⏳ Processing...")
        self.btn_browse.configure(state="disabled")
        self.status_label.configure(text="Running computer vision deskew routine...", text_color="#f1c40f")

        self.processing_thread = threading.Thread(target=self.execute_analysis_pipeline, daemon=True)
        self.processing_thread.start()

    def execute_analysis_pipeline(self):
        """Core asynchronous processing pipeline executor: handles CV + LLM engine stages."""
        start_time = time.time()
        filename = os.path.basename(self.current_image_path)
        logger.info(f"Starting analysis pipeline for file: {filename}")

        try:
            # Phase 1: Computer Vision geometric corrections
            logger.debug("Executing image orientation alignment & document localization pass...")
            processed_cv_matrix = deskew_and_crop(self.current_image_path)
            pil_ready_asset = opencv_to_pil(processed_cv_matrix)

            # Phase 2: AI / Large Language Model extraction pipeline parsing
            self.update_status_safe("Querying Groq Cloud endpoint API models...", "#f39c12")
            parser_engine = ReceiptParser()
            parsed_structured_receipt = parser_engine.parse_receipt_image(pil_ready_asset)

            elapsed = time.time() - start_time
            logger.info(f"Successfully processed receipt in {elapsed:.2f}s")

            # Queue back UI callback invocation onto the main event thread
            self.after(0, self.handle_pipeline_success, parsed_structured_receipt)

        except Exception as error_exception:
            logger.error(f"Critical execution crash during pipeline run: {error_exception}", exc_info=True)
            self.after(0, self.handle_pipeline_fault, str(error_exception))

    def update_status_safe(self, status_msg: str, text_hex_color: str):
        """Helper callback thread-safely updating the status label inside the window."""
        self.after(0, lambda: self.status_label.configure(text=status_msg, text_color=text_hex_color))

    def handle_pipeline_success(self, structured_receipt):
        """Callback on pipeline success. Prompts the user validation modal popup."""
        self.btn_scan.configure(state="normal", text="✨ Run OCR & Analysis")
        self.btn_browse.configure(state="normal")
        self.status_label.configure(text="Analysis complete!", text_color="#2ecc71")

        logger.debug("Launching interactive verification review sub-menu modal...")
        review_modal = ReceiptReviewWindow(self, structured_receipt, self.on_receipt_verified)
        review_modal.grab_set()  # Lock focus interaction onto child dialog exclusively

    def handle_pipeline_fault(self, system_error_details: str):
        """Gracefully rollback application interaction states when background tasks fail."""
        self.btn_scan.configure(state="normal", text="✨ Run OCR & Analysis")
        self.btn_browse.configure(state="normal")
        self.status_label.configure(text="Pipeline failure", text_color="#e74c3c")

        messagebox.showerror(
            "Pipeline Processing Error",
            f"An error occurred while analyzing the receipt:\n\n{system_error_details}\n\n"
            "Please check internet connections, API keys configurations or retry with a clearer image.",
        )

    def on_receipt_verified(self, final_verified_receipt):
        """Invoked when user confirms verification editor. Updates database/CSV storage."""
        try:
            logger.info("Saving user-validated structured invoice data back into local storage...")

            # Write transaction history into relational DB and CSV files
            insert_receipt(final_verified_receipt, self.current_image_path or "")
            append_items_to_csv(final_verified_receipt, self.current_image_path or "")

            # Re-sync list tables views on screen
            self.refresh_history_table()
            messagebox.showinfo("Success", "Receipt data successfully verified and saved to history!")

        except Exception as db_err:
            logger.error(f"Failed updating historical records storage layers: {db_err}", exc_info=True)
            messagebox.showerror("Storage Core Failure", f"Failed appending items to history database:\n{db_err}")

    def refresh_history_table(self):
        """Load items dataset from disk and format a plain-text table."""
        self.history_textbox.configure(state="normal")
        self.history_textbox.delete("1.0", "end")

        try:
            all_records = load_all_items()
            if not all_records:
                self.history_textbox.insert("1.0", "No scanned receipt items recorded yet in the global CSV file.")
                self.history_textbox.configure(state="disabled")
                return

            # Construct layout alignment header string labels
            table_header = f"{'DATE':<12} | {'STORE':<18} | {'ORIGINAL ITEM':<22} | {'TRANSLATION / EXPLANATION':<35} | {'CATEGORY':<15} | {'PRICE':<8}\n"
            divider = "-" * 122 + "\n"

            self.history_textbox.insert("end", table_header)
            self.history_textbox.insert("end", divider)

            for item in all_records:
                # Safely truncate overly broad text parameters to protect terminal column boundaries
                store = (item.get("store_name") or "Unknown")[:16]
                orig = (item.get("japanese_name") or "")[:20]
                trans = (item.get("english_name") or "")[:32]
                cat = (item.get("category") or "Other")[:13]

                row_line = (
                    f"{(item.get('date') or 'N/A'):<12} | "
                    f"{store:<18} | "
                    f"{orig:<22} | "
                    f"{trans:<35} | "
                    f"{cat:<15} | "
                    f"¥{safe_int(item.get('price')):<8}\n"
                )
                self.history_textbox.insert("end", row_line)

        except Exception as read_err:
            logger.error(f"Failed loading values inside history textbox UI panel: {read_err}")
            self.history_textbox.insert("1.0", f"Error rendering dataset storage tables:\n{read_err}")
        self.history_textbox.configure(state="disabled")

    # =========================================================================
    # GENERATE AND OPEN INTERACTIVE HTML DASHBOARD
    # =========================================================================
    def open_html_dashboard(self):
        """Generates a comprehensive HTML dashboard and launches it in the browser."""
        if generate_html_dashboard():
            if server_running:
                webbrowser.open("http://localhost:8000/")
            else:
                output_html_path = os.path.join(OUTPUT_DIR, "dashboard_kanji_kakei.html")
                webbrowser.open("file://" + os.path.realpath(output_html_path))
        else:
            messagebox.showinfo(
                "No Data Available", "No receipts scanned yet. Please upload and process a receipt first."
            )


def generate_html_dashboard() -> bool:
    """Generates a comprehensive HTML dashboard and saves it in OUTPUT_DIR.
    Returns True on success, False if no records are available.
    """
    try:
        records = load_all_items()
        if not records:
            return False

        # 1. Financial Analytics Compilation for Tab 2 (Global Recap)
        total_global = 0
        unique_receipts = len(set((r.get("date") or "") + (r.get("store_name") or "") for r in records))
        category_totals = defaultdict(float)

        for r in records:
            price = float(safe_int(r.get("price")))
            total_global += price
            cat = (r.get("category") or "Other").strip()
            category_totals[cat] += price

        # Determine largest spending cluster
        top_category = "None"
        top_category_amount = 0
        if category_totals:
            top_category = max(category_totals, key=category_totals.get)
            top_category_amount = category_totals[top_category]

        # Build HTML UI components for the category metrics bars
        categories_html = ""
        category_icons = {
            "Groceries": "🛒",
            "Drink": "🥤",
            "Snack": "🍿",
            "Dining Out": "🍜",
            "Daily Essentials": "🏠",
            "Clothes": "👕",
            "Personal Care": "💅",
            "Stationery": "✏️",
            "Leisure": "🎮",
            "Souvenirs": "🎁",
            "Tax": "💸",
            "Other": "📦",
        }
        category_colors = {
            "Groceries": "#0d47a1",
            "Drink": "#1565c0",
            "Snack": "#ff8f00",
            "Dining Out": "#e64a19",
            "Daily Essentials": "#2e7d32",
            "Clothes": "#00838f",
            "Personal Care": "#c2185b",
            "Stationery": "#6a1b9a",
            "Leisure": "#ad1457",
            "Souvenirs": "#ef6c00",
            "Tax": "#37474f",
            "Other": "#616161",
        }

        for cat in sorted_categories(category_totals.keys()):
            amount = category_totals[cat]
            percentage = (amount / total_global * 100) if total_global > 0 else 0
            icon = category_icons.get(cat, "📦")
            color = category_colors.get(cat, "#616161")
            categories_html += f"""
            <div class="category-progress">
                <div class="progress-header">
                    <span>{icon} {cat}</span>
                    <strong>¥{amount:,.0f} ({percentage:.1f}%)</strong>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar" style="width: {percentage}%; background-color: {color};"></div>
                </div>
            </div>
            """

        # 2. Extract Data Rows for Tab 1 (Most Recent Receipt Analysis Viewport)
        last_receipt_date = records[-1].get("date") or "Unknown"
        last_receipt_store = records[-1].get("store_name") or "Unknown"
        last_receipt_image_path = records[-1].get("image_path") or ""
        last_savings_advice = records[-1].get("savings_advice") or "No advice available."
        
        last_receipt_image_url = ""
        if last_receipt_image_path:
            last_receipt_image_url = "/receipts/" + os.path.basename(last_receipt_image_path)
            
        image_card_style = "" if last_receipt_image_url else "display: none;"
        receipt_items_html = ""
        last_receipt_total = 0

        # Match and safely bundle all entry items belonging to the same transaction
        for r in records:
            if (r.get("date") == records[-1].get("date")
                    and r.get("store_name") == records[-1].get("store_name")):
                price = float(safe_int(r.get("price")))
                last_receipt_total += price
                badge_class = (r.get("category") or "Other").lower().replace(" & ", "-").replace(" ", "-")

                receipt_items_html += f"""
                <tr>
                    <td><span class="jp-text">{r.get('japanese_name', '')}</span></td>
                    <td>{r.get('english_name', '')}</td>
                    <td><span class="badge badge-{badge_class}">{r.get('category', 'Other')}</span></td>
                    <td><strong>¥{price:,.0f}</strong></td>
                </tr>
                """

        # 3. Clean Responsive Component Embedded Core HTML Blueprint
        html_template = f"""<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Kanji-Kakei - Interactive Insights Dashboard</title>
            <style>
                :root {{
                    --primary: #2c3e50;
                    --primary-light: #34495e;
                    --accent: #1f6aa5;
                    --accent-hover: #2980b9;
                    --bg: #f8f9fa;
                    --card-bg: #ffffff;
                    --text: #2c3e50;
                    --text-muted: #7f8c8d;
                    --border: #e2e8f0;
                }}
                * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, sans-serif; }}
                body {{ background-color: var(--bg); color: var(--text); padding: 20px; }}
                .container {{ max-width: 1100px; margin: 0 auto; }}
                header {{ background-color: var(--card-bg); padding: 15px 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center; }}
                .logo {{ font-size: 1.4rem; font-weight: 700; color: var(--primary); }}
                .logo span {{ color: var(--accent); }}
                .nav-tabs {{ display: flex; gap: 10px; }}
                .tab-btn {{ background: none; border: none; padding: 10px 20px; font-size: 1rem; font-weight: 600; color: var(--text-muted); cursor: pointer; border-radius: 8px; transition: all 0.3s; }}
                .tab-btn:hover {{ color: var(--primary); background-color: #edf2f7; }}
                .tab-btn.active {{ color: #fff; background-color: var(--accent); }}
                .tab-content {{ display: none; }}
                .tab-content.active {{ display: block; animation: fadeIn 0.4s ease; }}
                @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
                
                .dashboard-grid {{ display: grid; grid-template-columns: 350px 1fr; gap: 25px; align-items: start; }}
                @media (max-width: 800px) {{ .dashboard-grid {{ grid-template-columns: 1fr; }} }}
                
                .card {{ position: relative; background: var(--card-bg); border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); padding: 25px; margin-bottom: 25px; border: 1px solid var(--border); }}
                .card-title {{ font-size: 1.2rem; font-weight: 600; margin-bottom: 20px; color: var(--primary); border-bottom: 2px solid var(--bg); padding-bottom: 10px; }}
                
                .dropzone {{ border: 2px dashed var(--accent); border-radius: 8px; background: #fdfdfd; padding: 30px 20px; text-align: center; cursor: pointer; transition: all 0.3s; margin-bottom: 10px; }}
                .dropzone:hover, .dropzone.dragover {{ background: #f0f7ff; border-color: var(--accent-hover); }}
                .dropzone-content {{ display: flex; flex-direction: column; align-items: center; gap: 10px; }}
                .upload-icon {{ font-size: 2rem; }}
                
                .preview-container {{ text-align: center; }}
                .preview-container img {{ max-width: 100%; max-height: 300px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }}
                .actions {{ display: flex; gap: 10px; justify-content: center; }}
                
                .btn {{ padding: 10px 20px; border-radius: 6px; border: none; font-weight: 600; cursor: pointer; font-size: 0.9rem; transition: all 0.2s; }}
                .btn-accent {{ background-color: var(--accent); color: white; }}
                .btn-accent:hover {{ background-color: var(--accent-hover); }}
                .btn-muted {{ background-color: #edf2f7; color: var(--text-muted); }}
                .btn-muted:hover {{ background-color: #e2e8f0; }}
                
                .processing-overlay {{ position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(255,255,255,0.85); backdrop-filter: blur(4px); z-index: 10; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 15px; border-radius: 12px; }}
                .spinner {{ border: 4px solid rgba(0,0,0,0.1); width: 50px; height: 50px; border-radius: 50%; border-left-color: var(--accent); animation: spin 1s linear infinite; }}
                @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
                .processing-status {{ font-weight: 600; color: var(--primary); }}
                .timer {{ font-size: 1.1rem; color: var(--text-muted); font-family: monospace; }}
                
                .receipt-image-wrapper {{ text-align: center; padding: 10px; }}
                .receipt-img {{ max-width: 100%; border-radius: 8px; max-height: 400px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }}
                
                .savings-advice-box {{ background-color: #fffde7; border-left: 4px solid #fbc02d; padding: 15px; border-radius: 0 8px 8px 0; margin-top: 20px; }}
                .advice-title {{ font-weight: 700; color: #f57f17; margin-bottom: 5px; font-size: 0.95rem; }}
                .advice-text {{ font-style: italic; color: #5d4037; font-size: 0.95rem; }}
                
                .receipt-summary {{ background: var(--bg); border-radius: 8px; padding: 15px; }}
                .meta-item {{ display: flex; justify-content: space-between; margin-bottom: 8px; }}
                .meta-item .label {{ color: var(--text-muted); }}
                .meta-item .value {{ font-weight: 600; }}
                .total-amount {{ font-size: 1.5rem; color: var(--accent); text-align: right; margin-top: 15px; font-weight: 700; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th {{ background-color: var(--bg); color: var(--primary); text-align: left; padding: 12px; font-weight: 600; }}
                td {{ padding: 14px 12px; border-bottom: 1px solid var(--border); font-size: 0.95rem; }}
                .jp-text {{ font-family: 'Hiragino Kaku Gothic Pro', 'Meiryo', sans-serif; font-weight: bold; color: #d35400; background: #fff5eb; padding: 2px 6px; border-radius: 4px; }}
                .badge {{ display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; color: #fff; }}
                .badge-groceries {{ background-color: #0d47a1; }}
                .badge-drink {{ background-color: #1565c0; }}
                .badge-snack {{ background-color: #ff8f00; }}
                .badge-dining-out {{ background-color: #e64a19; }}
                .badge-daily-essentials {{ background-color: #2e7d32; }}
                .badge-clothes {{ background-color: #00838f; }}
                .badge-personal-care {{ background-color: #c2185b; }}
                .badge-stationery {{ background-color: #6a1b9a; }}
                .badge-leisure {{ background-color: #ad1457; }}
                .badge-souvenirs {{ background-color: #ef6c00; }}
                .badge-tax {{ background-color: #37474f; }}
                .badge-other {{ background-color: #616161; }}
                .recap-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
                .stat-card {{ background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%); color: white; padding: 20px; border-radius: 12px; }}
                .stat-card.accent-card {{ background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%); }}
                .stat-label {{ font-size: 0.9rem; opacity: 0.8; }}
                .stat-value {{ font-size: 1.8rem; font-weight: 700; margin-top: 5px; }}
                .category-progress {{ margin-bottom: 20px; }}
                .progress-header {{ display: flex; justify-content: space-between; margin-bottom: 6px; }}
                .progress-bar-container {{ background-color: var(--border); height: 10px; border-radius: 5px; overflow: hidden; }}
                .progress-bar {{ height: 100%; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <div class="logo">🇯🇵 Kanji-Kakei <span>. Dashboard</span></div>
                    <div class="nav-tabs">
                        <button class="tab-btn active" onclick="switchTab('analysis')">Receipt Analysis</button>
                        <button class="tab-btn" onclick="switchTab('recap')">Monthly Summary Breakdown</button>
                    </div>
                </header>

                <main id="analysis" class="tab-content active">
                    <div class="dashboard-grid">
                        <!-- Left Column: Upload and Preview -->
                        <div class="grid-col-left">
                            <div class="card">
                                <div class="card-title">📸 Upload & Process Receipt</div>
                                <div id="dropzone" class="dropzone">
                                    <div class="dropzone-content">
                                        <span class="upload-icon">📥</span>
                                        <p>Drag & drop receipt image or <strong>browse</strong></p>
                                        <input type="file" id="fileInput" accept="image/*" style="display: none;" />
                                    </div>
                                </div>
                                
                                <div id="preview-container" class="preview-container" style="display: none;">
                                    <img id="image-preview" src="" alt="Receipt Preview" />
                                    <div class="actions">
                                        <button id="btn-process" class="btn btn-accent">⚡ Process Receipt</button>
                                        <button id="btn-cancel" class="btn btn-muted">Cancel</button>
                                    </div>
                                </div>
                                
                                <div id="processing-overlay" class="processing-overlay" style="display: none;">
                                    <div class="spinner"></div>
                                    <div class="processing-status">Analyzing receipt content...</div>
                                    <div class="timer">Elapsed: <span id="timer-val">0.0</span>s</div>
                                </div>
                            </div>
                            
                            <div class="card" style="{image_card_style}">
                                <div class="card-title">Processed Image Viewport</div>
                                <div class="receipt-image-wrapper">
                                    <img src="{last_receipt_image_url}" alt="Last Processed Receipt" class="receipt-img" />
                                </div>
                            </div>
                        </div>
                        
                        <!-- Right Column: Results -->
                        <div class="grid-col-right">
                            <div class="card">
                                <div class="card-title">Latest Receipt Metadata</div>
                                <div class="receipt-summary">
                                    <div class="meta-item"><span class="label">Store Location</span><span class="value">{last_receipt_store}</span></div>
                                    <div class="meta-item"><span class="label">Transaction Date</span><span class="value">{last_receipt_date}</span></div>
                                    <div class="total-amount">Invoice Total: ¥{last_receipt_total:,.0f}</div>
                                </div>
                                <div class="savings-advice-box">
                                    <div class="advice-title">💡 Financial Advice</div>
                                    <p class="advice-text">"{last_savings_advice}"</p>
                                </div>
                            </div>
                            <div class="card">
                                <div class="card-title">Extracted & Interpreted Line Items</div>
                                <div style="overflow-x: auto;">
                                    <table>
                                        <thead>
                                            <tr><th>Japanese Raw OCR</th><th>Translation / Context</th><th>Category</th><th>Price</th></tr>
                                        </thead>
                                        <tbody>
                                            {receipt_items_html}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </main>

                <main id="recap" class="tab-content">
                    <div class="recap-grid">
                        <div class="stat-card"><div class="stat-label">Total Cumulative Expenses</div><div class="stat-value">¥{total_global:,.0f}</div></div>
                        <div class="stat-card accent-card"><div class="stat-label">Processed Receipts Count</div><div class="stat-value">{unique_receipts}</div></div>
                        <div class="stat-card" style="background: #27ae60;"><div class="stat-label">Top Budget Allocation</div><div class="stat-value" style="font-size: 1.3rem;">{top_category} (¥{top_category_amount:,.0f})</div></div>
                    </div>
                    <div class="card">
                        <div class="card-title">Budget Weight Distribution by Category</div>
                        {categories_html}
                    </div>
                </main>
            </div>

            <script>
                function switchTab(tabId) {{
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                    document.getElementById(tabId).classList.add('active');
                    event.currentTarget.classList.add('active');
                }}
                
                const dropzone = document.getElementById('dropzone');
                const fileInput = document.getElementById('fileInput');
                const previewContainer = document.getElementById('preview-container');
                const imagePreview = document.getElementById('image-preview');
                const btnProcess = document.getElementById('btn-process');
                const btnCancel = document.getElementById('btn-cancel');
                const processingOverlay = document.getElementById('processing-overlay');
                const timerVal = document.getElementById('timer-val');
                let selectedFile = null;
                let timerInterval = null;

                dropzone.addEventListener('click', () => fileInput.click());
                dropzone.addEventListener('dragover', (e) => {{
                    e.preventDefault();
                    dropzone.classList.add('dragover');
                }});
                dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
                dropzone.addEventListener('drop', (e) => {{
                    e.preventDefault();
                    dropzone.classList.remove('dragover');
                    if (e.dataTransfer.files.length > 0) {{
                        handleFile(e.dataTransfer.files[0]);
                    }}
                }});

                fileInput.addEventListener('change', (e) => {{
                    if (e.target.files.length > 0) {{
                        handleFile(e.target.files[0]);
                    }}
                }});

                function handleFile(file) {{
                    if (!file.type.startsWith('image/')) {{
                        alert('Please select an image file.');
                        return;
                    }}
                    selectedFile = file;
                    const reader = new FileReader();
                    reader.onload = (e) => {{
                        imagePreview.src = e.target.result;
                        dropzone.style.display = 'none';
                        previewContainer.style.display = 'block';
                    }};
                    reader.readAsDataURL(file);
                }}

                btnCancel.addEventListener('click', () => {{
                    selectedFile = null;
                    dropzone.style.display = 'block';
                    previewContainer.style.display = 'none';
                    fileInput.value = '';
                }});

                btnProcess.addEventListener('click', () => {{
                    if (!selectedFile) return;
                    
                    processingOverlay.style.display = 'flex';
                    let startTime = Date.now();
                    timerInterval = setInterval(() => {{
                        let elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                        timerVal.textContent = elapsed;
                    }}, 100);
                    
                    const reader = new FileReader();
                    reader.onload = () => {{
                        const payload = {{
                            filename: selectedFile.name,
                            image: reader.result
                        }};
                        
                        fetch('/api/upload', {{
                            method: 'POST',
                            headers: {{
                                'Content-Type': 'application/json'
                            }},
                            body: JSON.stringify(payload)
                        }})
                        .then(res => res.json())
                        .then(data => {{
                            clearInterval(timerInterval);
                            if (data.success) {{
                                window.location.reload();
                            }} else {{
                                processingOverlay.style.display = 'none';
                                alert('Analysis failed: ' + (data.error || 'Unknown error'));
                            }}
                        }})
                        .catch(err => {{
                            clearInterval(timerInterval);
                            processingOverlay.style.display = 'none';
                            alert('Server error: ' + err.message);
                        }});
                    }};
                    reader.readAsDataURL(selectedFile);
                }});
            </script>
        </body>
        </html>
        """

        # Save the runtime generated document onto the outputs directory
        output_html_path = os.path.join(OUTPUT_DIR, "dashboard_kanji_kakei.html")
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_template)

        return True

    except Exception as err:
        logger.error(f"Dashboard assembly module encountered a runtime error: {err}", exc_info=True)
        return False


# =========================================================================
# INNER MODAL SUB-MENU DIALOG CLASS IMPLEMENTATION
# =========================================================================
class ReceiptReviewWindow(ctk.CTkToplevel):
    def __init__(self, parent_window, raw_receipt_data, on_save_callback):
        super().__init__(parent_window)

        self.parent_window = parent_window
        self.on_save_callback = on_save_callback

        # De-serialize a working data copy to avoid editing the payload directly.
        # NOTE: keys here match the pipeline's schema (store_name / total_amount /
        # tax_amount), which is what ReceiptParser actually returns.
        self.receipt_meta = {
            "store_name": raw_receipt_data.get("store_name", "Unknown Store"),
            "date": raw_receipt_data.get("date", datetime.now().strftime("%Y-%m-%d")),
            "total_amount": raw_receipt_data.get("total_amount", 0),
            "tax_amount": raw_receipt_data.get("tax_amount", 0),
        }
        self.items_list = list(raw_receipt_data.get("items", []))

        # Setup configuration bindings
        self.title("Verify Parsed Receipt Contents")
        self.geometry("900x600")
        self.minsize(800, 500)

        # Build dynamic grid constraints
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Render sections
        self.render_header_banner()
        self.render_left_metadata_fields()
        self.render_right_items_table()
        self.render_bottom_action_bar()

    def render_header_banner(self):
        """Build top contextual layout row providing quick guidance to the user."""
        self.lbl_header = ctk.CTkLabel(
            self,
            text="Double check data entries extracted by the LLM below before saving:",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#3498db",
            anchor="w",
        )
        self.lbl_header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(15, 10))

    def render_left_metadata_fields(self):
        """Generate interactive widgets allowing rapid corrections to global transaction fields."""
        self.left_panel = ctk.CTkFrame(self, corner_radius=8)
        self.left_panel.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=10)

        ctk.CTkLabel(self.left_panel, text="Global Receipt Metadata", font=ctk.CTkFont(size=14, weight="bold")).pack(
            padx=15, pady=(15, 10), anchor="w"
        )

        # Field 1: Store name string title
        ctk.CTkLabel(self.left_panel, text="Store Name (Original/Translated):").pack(padx=15, pady=(5, 0), anchor="w")
        self.ent_store = ctk.CTkEntry(self.left_panel)
        self.ent_store.pack(padx=15, pady=(0, 10), fill="x")
        self.ent_store.insert(0, self.receipt_meta["store_name"])

        # Field 2: Transaction timestamp date string anchor
        ctk.CTkLabel(self.left_panel, text="Transaction Date (YYYY-MM-DD):").pack(padx=15, pady=(5, 0), anchor="w")
        self.ent_date = ctk.CTkEntry(self.left_panel)
        self.ent_date.pack(padx=15, pady=(0, 10), fill="x")
        self.ent_date.insert(0, self.receipt_meta["date"])

        # Field 3: Total value tracking number integer
        ctk.CTkLabel(self.left_panel, text="Grand Total Price (¥):").pack(padx=15, pady=(5, 0), anchor="w")
        self.ent_total = ctk.CTkEntry(self.left_panel)
        self.ent_total.pack(padx=15, pady=(0, 10), fill="x")
        self.ent_total.insert(0, str(self.receipt_meta["total_amount"]))

        # Field 4: Internal consumption tax reference estimation
        ctk.CTkLabel(self.left_panel, text="Included Duty Taxes (¥):").pack(padx=15, pady=(5, 0), anchor="w")
        self.ent_taxes = ctk.CTkEntry(self.left_panel)
        self.ent_taxes.pack(padx=15, pady=(0, 15), fill="x")
        self.ent_taxes.insert(0, str(self.receipt_meta["tax_amount"]))

    def render_right_items_table(self):
        """Embed an editor panel mapping rows for each parsed product."""
        self.right_panel = ctk.CTkFrame(self, corner_radius=8)
        self.right_panel.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=10)

        ctk.CTkLabel(self.right_panel, text="Extracted Product Line Items", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, columnspan=4, padx=15, pady=(15, 10), sticky="w"
        )

        # Generate table header cell labels
        headers = ["Original Item Text", "Translation / Meaning", "Category Choice", "Price"]
        for idx, col_title in enumerate(headers):
            lbl = ctk.CTkLabel(self.right_panel, text=col_title, font=ctk.CTkFont(size=11, weight="bold"), text_color="#888888")
            lbl.grid(row=1, column=idx, padx=8, pady=2, sticky="w")

        # Scrollable inner row content panel container
        self.scroll_table = ctk.CTkScrollableFrame(self.right_panel, fg_color="transparent")
        self.scroll_table.grid(row=2, column=0, columnspan=4, sticky="nsew", padx=5, pady=5)
        self.right_panel.grid_rowconfigure(2, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=2)
        self.right_panel.grid_columnconfigure(1, weight=2)
        self.right_panel.grid_columnconfigure(2, weight=1)
        self.right_panel.grid_columnconfigure(3, weight=1)

        # Allocate and fill dynamic cell rows matching the input structure.
        # NOTE: item fields here match the pipeline's schema (japanese_name /
        # english_name), which is what was previously read with the wrong keys
        # and caused the sub-window text to be blank.
        self.row_widget_bindings = []
        for index, item in enumerate(self.items_list):
            # Column 0: Original OCR (Japanese) text
            entry_ocr = ctk.CTkEntry(self.scroll_table, font=ctk.CTkFont(size=12))
            entry_ocr.grid(row=index, column=0, padx=4, pady=4, sticky="ew")
            entry_ocr.insert(0, item.get("japanese_name", ""))

            # Column 1: English translation
            entry_trans = ctk.CTkEntry(self.scroll_table, font=ctk.CTkFont(size=12))
            entry_trans.grid(row=index, column=1, padx=4, pady=4, sticky="ew")
            entry_trans.insert(0, item.get("english_name", ""))

            # Column 2: Category option menu
            opt_cat = ctk.CTkOptionMenu(self.scroll_table, values=CATEGORY_ORDER, font=ctk.CTkFont(size=11))
            opt_cat.grid(row=index, column=2, padx=4, pady=4, sticky="ew")
            current_cat = (item.get("category") or "Other").strip()
            if current_cat not in CATEGORY_ORDER:
                current_cat = "Other"
            opt_cat.set(current_cat)

            # Column 3: Price
            entry_price = ctk.CTkEntry(self.scroll_table, width=65, font=ctk.CTkFont(size=12))
            entry_price.grid(row=index, column=3, padx=4, pady=4, sticky="ew")
            entry_price.insert(0, str(item.get("price", 0)))

            self.scroll_table.grid_columnconfigure(0, weight=2)
            self.scroll_table.grid_columnconfigure(1, weight=2)
            self.scroll_table.grid_columnconfigure(2, weight=1)
            self.scroll_table.grid_columnconfigure(3, weight=1)

            # Keep widget handle references in an active memory list
            self.row_widget_bindings.append(
                {"ocr": entry_ocr, "translation": entry_trans, "category": opt_cat, "price": entry_price}
            )

    def render_bottom_action_bar(self):
        """Construct lower button controls grid row wrapper panel."""
        self.bottom_bar = ctk.CTkFrame(self, height=60, fg_color="transparent")
        self.bottom_bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=20, pady=(10, 20))

        self.btn_save = ctk.CTkButton(
            self.bottom_bar, text="💾 Confirm & Save to History", width=220, height=38, command=self.on_confirm_save
        )
        self.btn_save.pack(side="right", padx=10)

        self.btn_abort = ctk.CTkButton(
            self.bottom_bar,
            text="❌ Discard Scanned Data",
            width=160,
            height=38,
            fg_color="#c0392b",
            hover_color="#962d22",
            command=self.on_cancel,
        )
        self.btn_abort.pack(side="left", padx=10)

    def on_confirm_save(self):
        """Fetch updated field values, sanitize types, and emit a receipt dict in the
        pipeline's schema so insert_receipt() and append_items_to_csv() store it correctly."""
        try:
            # 1. Collect global fields, keyed to match the DB/CSV layers
            validated_receipt = {
                "store_name": self.ent_store.get().strip() or "Unknown Store",
                "date": self.ent_date.get().strip() or datetime.now().strftime("%Y-%m-%d"),
                "total_amount": int(self.ent_total.get().strip() or 0),
                "tax_amount": int(self.ent_taxes.get().strip() or 0),
                "items": [],
            }

            # 2. Pull each item row's values
            for binding in self.row_widget_bindings:
                try:
                    price_val = int(binding["price"].get().strip() or 0)
                except ValueError:
                    price_val = 0

                item_row = {
                    "japanese_name": binding["ocr"].get().strip(),
                    "english_name": binding["translation"].get().strip(),
                    "category": binding["category"].get(),
                    "price": price_val,
                }
                validated_receipt["items"].append(item_row)

            logger.debug(
                f"Validation pass generated {len(validated_receipt['items'])} item rows."
            )

            # Fire execution callback back to the master window controller
            self.on_save_callback(validated_receipt)
            self.destroy()

        except ValueError as parse_format_err:
            logger.warning(f"User submitted un-parseable numeric value: {parse_format_err}")
            messagebox.showerror(
                "Format Input Error",
                f"Please verify total price and tax values are whole integers:\n{parse_format_err}",
            )
        except Exception as general_err:
            logger.error(f"Failed marshalling data validation structures: {general_err}")

    def on_cancel(self):
        """Close the sub-menu without saving anything."""
        logger.info("User cancelled the receipt review. Discarding scanned data.")
        self.destroy()


# Server implementation and execution gateway
server_running = False

class DashboardHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.getcwd(), **kwargs)

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(302)
            self.send_header('Location', '/outputs/dashboard_kanji_kakei.html')
            self.end_headers()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/api/upload':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                # Parse JSON
                data = json.loads(post_data.decode('utf-8'))
                filename = data.get("filename", "uploaded_receipt.jpg")
                image_data_url = data.get("image", "")
                
                if not image_data_url or "," not in image_data_url:
                    raise ValueError("Invalid image data URL format")
                
                # Decode base64
                header, base64_data = image_data_url.split(',', 1)
                image_bytes = base64.b64decode(base64_data)
                
                # Save image
                os.makedirs("receipts", exist_ok=True)
                name, ext = os.path.splitext(filename)
                unique_filename = f"{name}_{int(time.time())}{ext}"
                save_path = os.path.join("receipts", unique_filename)
                
                with open(save_path, "wb") as f:
                    f.write(image_bytes)
                
                logger.info(f"[Server] Saved uploaded receipt to: {save_path}")
                
                # Preprocess image
                try:
                    deskewed_np = deskew_and_crop(save_path)
                    pil_img = opencv_to_pil(deskewed_np)
                except Exception as e:
                    logger.warning(f"[Server] Preprocessing failed: {e}. Using original image.")
                    pil_img = Image.open(save_path).convert("RGB")
                
                # Run LLM parsing
                parser = ReceiptParser()
                response = parser.parse_receipt_image(pil_img)
                
                # Insert into DB and CSV
                insert_receipt(response, save_path)
                append_items_to_csv(response, save_path)
                
                # Regenerate dashboard HTML
                generate_html_dashboard()
                
                # Trigger GUI refresh if active
                try:
                    if 'app' in globals() and app:
                        app.after(0, app.refresh_history_table)
                except Exception as gui_err:
                    logger.debug(f"Could not refresh GUI: {gui_err}")
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                success_response = {
                    "success": True,
                    "data": response
                }
                self.wfile.write(json.dumps(success_response).encode('utf-8'))
                
            except Exception as e:
                logger.error(f"[Server] API processing error: {e}", exc_info=True)
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_response = {
                    "success": False,
                    "error": str(e)
                }
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def start_web_server():
    server_address = ('', 8000)
    class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        allow_reuse_address = True
        
    def run_server():
        global server_running
        try:
            httpd = ThreadingHTTPServer(server_address, DashboardHTTPRequestHandler)
            server_running = True
            logger.info("[Server] Local dashboard web server running on http://localhost:8000/")
            httpd.serve_forever()
        except Exception as e:
            logger.error(f"[Server] Failed to start local web server: {e}")
            
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

# Main execution gateway
if __name__ == "__main__":
    logger.info("Starting Kanji-Kakei main graphical window lifecycle...")
    start_web_server()
    app = ReceiptApp()
    app.mainloop()
