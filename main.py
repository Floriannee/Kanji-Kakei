import os
import time
import logging
import threading
from datetime import datetime
from collections import defaultdict

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
CATEGORY_ORDER = ["Food", "Drinks", "Snacks", "Household", "Stationery", "Health & Beauty", "Other"]


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


class KanjiKakeiApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure Window
        self.title("漢字家計 | Kanji-Kakei Receipt Reader")
        self.geometry("1100x700")
        self.minsize(950, 600)

        # Initialize Core components
        self.parser = ReceiptParser()
        init_db()    # Ensure SQLite tables exist (kept for compatibility with existing tooling)
        init_csv()   # Ensure the continually-updated CSV ledger exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Variables
        self.uploaded_image_path = None
        self.processed_pil_image = None
        self.start_time = None
        self.is_processing = False

        # Build UI Elements
        self.init_ui()
        self.refresh_main_list()
        logger.info("Application interface initialized successfully.")

    def init_ui(self):
        # Configure Grid Layout (1 row, 2 columns: Sidebar & Main Area)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # ----------------- SIDEBAR PANEL -----------------
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

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

        # Single combined Upload + Scan button. Confirming the resulting review
        # sub-menu is what actually commits (and "exports") the data to the CSV ledger.
        self.upload_btn = ctk.CTkButton(
            self.sidebar_frame,
            text="📁 Upload Receipt",
            command=self.upload_and_scan_receipt,
            font=ctk.CTkFont(weight="bold")
        )
        self.upload_btn.grid(row=2, column=0, padx=20, pady=10)

        # Real-time Stopwatch Label (active only while a receipt is being scanned)
        self.stopwatch_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="",
            font=ctk.CTkFont(size=14, weight="bold", family="Courier New"),
            text_color="lightblue"
        )
        self.stopwatch_label.grid(row=3, column=0, padx=20, pady=10)

        # Running totals for everything currently stored in the CSV ledger
        self.summary_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            justify="left"
        )
        self.summary_label.grid(row=5, column=0, padx=20, pady=(10, 20), sticky="s")

        # ----------------- MAIN VIEW AREA -----------------
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        self.main_header = ctk.CTkLabel(
            self.main_frame,
            text="Stored Receipts — By Category",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.main_header.grid(row=0, column=0, padx=10, pady=(0, 10), sticky="w")

        # Scrollable categorized list, populated from the CSV ledger
        self.list_scroll_frame = ctk.CTkScrollableFrame(self.main_frame, fg_color="transparent")
        self.list_scroll_frame.grid(row=1, column=0, sticky="nsew")
        self.list_scroll_frame.grid_columnconfigure(0, weight=1)

    # ----------------- Main category list (CSV-backed) -----------------

    def refresh_main_list(self):
        """Reload every stored item from the CSV ledger and rebuild the categorized list view."""
        items = load_all_items()
        self.render_category_list(
            self.list_scroll_frame,
            items,
            empty_message="No receipts uploaded yet.\nClick 'Upload Receipt' to start."
        )
        self.update_summary(items)

    def update_summary(self, items):
        total_spent = sum(safe_int(it.get("price")) for it in items)
        self.summary_label.configure(text=f"{len(items)} item(s) tracked\nTotal: ¥{total_spent:,}")

    def render_category_list(self, container, items, empty_message="Nothing to show yet."):
        """
        Render `items` (a list of dicts with at least japanese_name, english_name,
        category, price, store_name, date) into `container`, grouped under category
        headers. Used for both the main window list and the review sub-menu list.
        """
        for widget in container.winfo_children():
            widget.destroy()

        if not items:
            placeholder = ctk.CTkLabel(container, text=empty_message, text_color="gray", justify="center")
            placeholder.grid(row=0, column=0, padx=10, pady=40)
            return

        grouped = defaultdict(list)
        for it in items:
            grouped[(it.get("category") or "Other").strip() or "Other"].append(it)

        row_idx = 0
        for category in sorted_categories(grouped.keys()):
            cat_items = grouped[category]
            cat_total = sum(safe_int(it.get("price")) for it in cat_items)

            cat_label = ctk.CTkLabel(
                container,
                text=f"{category}  ·  {len(cat_items)} item(s)  ·  ¥{cat_total:,}",
                font=ctk.CTkFont(size=14, weight="bold"),
                anchor="w"
            )
            cat_label.grid(row=row_idx, column=0, sticky="ew", padx=5, pady=(15 if row_idx else 0, 5))
            row_idx += 1

            for it in cat_items:
                row_frame = ctk.CTkFrame(container, fg_color=("gray85", "gray20"))
                row_frame.grid(row=row_idx, column=0, sticky="ew", padx=5, pady=2)
                row_frame.grid_columnconfigure(0, weight=1)

                name_text = f"{it.get('japanese_name', '')}  ({it.get('english_name', '')})"
                detail_text = f"{it.get('store_name', '')}  ·  {it.get('date', '')}"

                name_label = ctk.CTkLabel(
                    row_frame, text=name_text, anchor="w", font=ctk.CTkFont(size=12, weight="bold")
                )
                name_label.grid(row=0, column=0, sticky="w", padx=10, pady=(6, 0))

                detail_label = ctk.CTkLabel(
                    row_frame, text=detail_text, anchor="w", font=ctk.CTkFont(size=10), text_color="gray"
                )
                detail_label.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 6))

                price_label = ctk.CTkLabel(
                    row_frame,
                    text=f"¥{safe_int(it.get('price')):,}",
                    font=ctk.CTkFont(size=13, weight="bold"),
                    text_color="lightblue"
                )
                price_label.grid(row=0, column=1, rowspan=2, padx=10, sticky="e")

                row_idx += 1

    # ----------------- Upload + Scan flow -----------------

    def upload_and_scan_receipt(self):
        """Open a file dialog, preprocess the image, then kick off background scanning."""
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
        except Exception as err:
            logger.error(f"Failed to load and preprocess receipt image: {err}")
            messagebox.showerror("Error", f"Failed to load receipt image: {err}")
            return

        self.start_scanning()

    def start_scanning(self):
        """Begin the background inference call and start the stopwatch."""
        self.is_processing = True
        self.upload_btn.configure(state="disabled", text="⏳ Scanning...")

        self.start_time = time.time()
        self.tick_stopwatch()

        # Execute multimodal API pipeline in background thread to prevent GUI lockup
        thread = threading.Thread(target=self.run_inference, args=(self.processed_pil_image,))
        thread.daemon = True
        thread.start()

    def tick_stopwatch(self):
        """Updates the elapsed time on the GUI every 50ms while scanning is active."""
        if self.is_processing and self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.stopwatch_label.configure(text=f"Scanning: {elapsed:.2f}s")
            self.after(50, self.tick_stopwatch)

    def run_inference(self, pil_img: Image.Image):
        """Runs the API parser client. Executes in the background thread."""
        try:
            result = self.parser.parse_receipt_image(pil_img)
            self.after(0, self.on_inference_complete, result)
        except Exception as e:
            logger.error(f"Inference process failed: {e}")
            self.after(0, self.on_inference_failed, str(e))

    def on_inference_complete(self, result: dict):
        """Callback on the main GUI thread once scanning finishes: opens the review sub-menu."""
        self.is_processing = False
        elapsed = time.time() - self.start_time
        self.stopwatch_label.configure(text=f"Scanned in {elapsed:.2f}s")
        self.upload_btn.configure(state="normal", text="📁 Upload Receipt")

        ReceiptReviewDialog(self, result, self.processed_pil_image, self.uploaded_image_path)

    def on_inference_failed(self, error_message: str):
        """Callback on the main GUI thread if scanning fails."""
        self.is_processing = False
        self.stopwatch_label.configure(text="Scan failed")
        self.upload_btn.configure(state="normal", text="📁 Upload Receipt")

        messagebox.showerror(
            "API Inference Error",
            f"The receipt scan failed.\n\nPlease check your internet connection or verify your API key.\n\nDetails: {error_message}"
        )

    # ----------------- Confirm callback from the review sub-menu -----------------

    def confirm_receipt(self, result: dict, image_path: str) -> bool:
        """
        Called by ReceiptReviewDialog when the user presses Confirm.
        Commits the scanned data to the CSV ledger (and SQLite for compatibility),
        then refreshes the main category list. Returns True on success.
        """
        db_success = True
        try:
            insert_receipt(result, image_path or "")
        except Exception as db_err:
            db_success = False
            logger.error(f"SQLite save failed: {db_err}")

        try:
            append_items_to_csv(result, image_path or "")
        except Exception as csv_err:
            logger.error(f"CSV save failed: {csv_err}")
            messagebox.showerror(
                "Storage Error",
                f"Failed to save receipt data to the CSV ledger.\n\nDetails: {csv_err}"
            )
            return False

        self.refresh_main_list()

        if db_success:
            messagebox.showinfo("Saved", "Receipt data has been added to your records.")
        else:
            messagebox.showwarning(
                "Partial Save",
                "Receipt was saved to the CSV ledger, but the local database backup failed."
            )
        return True


class ReceiptReviewDialog(ctk.CTkToplevel):
    """
    Sub-menu shown right after a receipt has been scanned. Lets the user inspect
    the scanned image and the parsed line items before committing anything.
    Cancel discards everything; Confirm writes the data to the CSV ledger.
    """

    def __init__(self, parent: KanjiKakeiApp, result: dict, pil_image: Image.Image, image_path: str):
        super().__init__(parent)
        self.parent_app = parent
        self.result = result
        self.pil_image = pil_image
        self.image_path = image_path

        self.title("Review Scanned Receipt")
        self.geometry("950x650")
        self.minsize(800, 550)
        self.transient(parent)
        self.grab_set()  # modal: block interaction with the main window until closed

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # ----- Left: scanned image preview -----
        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        self.image_frame.grid_rowconfigure(1, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.image_frame, text="Scanned Image", font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, pady=(10, 5))

        self.image_label = ctk.CTkLabel(self.image_frame, text="")
        self.image_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        # Defer rendering until the frame has been laid out and has real dimensions
        self.after(50, self.render_preview_image)

        # ----- Right: parsed line items, grouped by category -----
        self.info_frame = ctk.CTkFrame(self)
        self.info_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        self.info_frame.grid_rowconfigure(2, weight=1)
        self.info_frame.grid_columnconfigure(0, weight=1)

        store_text = result.get("store_name", "Unknown Store")
        ctk.CTkLabel(
            self.info_frame, text=store_text, font=ctk.CTkFont(size=16, weight="bold"), anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))

        totals_text = f"Total: ¥{safe_int(result.get('total_amount')):,}   ·   Tax: ¥{safe_int(result.get('tax_amount')):,}"
        ctk.CTkLabel(
            self.info_frame, text=totals_text, font=ctk.CTkFont(size=12), text_color="gray", anchor="w"
        ).grid(row=1, column=0, sticky="w", padx=10, pady=(0, 10))

        self.items_scroll = ctk.CTkScrollableFrame(self.info_frame, fg_color="transparent")
        self.items_scroll.grid(row=2, column=0, sticky="nsew", padx=5)
        self.items_scroll.grid_columnconfigure(0, weight=1)

        display_items = self._build_display_items(result)
        # Reuse the exact same renderer as the main window so the formats match.
        parent.render_category_list(
            self.items_scroll, display_items, empty_message="No items detected on this receipt."
        )

        if result.get("savings_advice"):
            advice_label = ctk.CTkLabel(
                self.info_frame,
                text=f"💡 {result['savings_advice']}",
                wraplength=380,
                justify="left",
                text_color="lightgreen"
            )
            advice_label.grid(row=3, column=0, sticky="w", padx=10, pady=10)

        # ----- Bottom action buttons -----
        self.button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.button_frame.grid(row=1, column=0, columnspan=2, pady=(0, 15))

        self.cancel_btn = ctk.CTkButton(
            self.button_frame,
            text="✖ Cancel",
            fg_color="gray40",
            hover_color="gray30",
            command=self.on_cancel
        )
        self.cancel_btn.grid(row=0, column=0, padx=10)

        self.confirm_btn = ctk.CTkButton(
            self.button_frame,
            text="✔ Confirm & Save",
            fg_color="green",
            hover_color="darkgreen",
            command=self.on_confirm
        )
        self.confirm_btn.grid(row=0, column=1, padx=10)

    @staticmethod
    def _build_display_items(result: dict) -> list:
        """Reshape the raw parsed receipt into the row format render_category_list expects."""
        store_name = result.get("store_name", "Unknown")
        date_str = datetime.now().strftime("%Y-%m-%d")
        display_items = []
        for it in result.get("items", []):
            display_items.append({
                "japanese_name": it.get("japanese_name", ""),
                "english_name": it.get("english_name", ""),
                "category": it.get("category") or "Other",
                "price": it.get("price", 0),
                "store_name": store_name,
                "date": date_str,
            })
        return display_items

    def render_preview_image(self):
        """Scale and show the scanned image inside the sub-menu's preview frame."""
        frame_width = self.image_frame.winfo_width()
        frame_height = self.image_frame.winfo_height()

        # Safety fallback if widget dimensions aren't initialized yet
        if frame_width < 100:
            frame_width = 420
        if frame_height < 100:
            frame_height = 500

        img_width, img_height = self.pil_image.size
        ratio = min((frame_width - 40) / img_width, (frame_height - 80) / img_height)
        ratio = max(ratio, 0.05)

        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        ctk_img = ctk.CTkImage(light_image=self.pil_image, dark_image=self.pil_image, size=(new_width, new_height))
        self.image_label.configure(image=ctk_img, text="")
        self.image_label.image = ctk_img  # Keep reference

    def on_cancel(self):
        """Close the sub-menu without saving anything."""
        logger.info("User cancelled the receipt review. Discarding scanned data.")
        self.destroy()

    def on_confirm(self):
        """Commit the scanned data to the CSV ledger via the parent app, then close."""
        success = self.parent_app.confirm_receipt(self.result, self.image_path)
        if success:
            self.destroy()


if __name__ == "__main__":
    app = KanjiKakeiApp()
    app.mainloop()
