import unittest
import os
import sqlite3
import numpy as np
import cv2
from PIL import Image
from config.settings import DB_FILE
from database.db_manager import init_db, insert_receipt, get_connection
from utils.image_processing import order_points, deskew_and_crop, opencv_to_pil
from inference.pipeline import ReceiptParser

class TestCoreComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize test databases
        init_db()

    def test_order_points(self):
        # 4 points in arbitrary order
        pts = np.array([
            [100, 100],  # top-left
            [200, 100],  # top-right
            [200, 200],  # bottom-right
            [100, 200]   # bottom-left
        ], dtype="float32")
        
        ordered = order_points(pts)
        
        # Verify ordering [top-left, top-right, bottom-right, bottom-left]
        np.testing.assert_array_almost_equal(ordered[0], [100, 100])
        np.testing.assert_array_almost_equal(ordered[1], [200, 100])
        np.testing.assert_array_almost_equal(ordered[2], [200, 200])
        np.testing.assert_array_almost_equal(ordered[3], [100, 200])

    def test_deskew_fallback(self):
        # Create a dummy blank image
        dummy_img_path = "dummy_test_image.png"
        img = np.zeros((300, 300, 3), dtype="uint8")
        cv2.imwrite(dummy_img_path, img)
        
        try:
            # Run deskew, which should fail contour detection and fall back to original image
            preprocessed = deskew_and_crop(dummy_img_path)
            self.assertEqual(preprocessed.shape, (300, 300, 3))
        finally:
            if os.path.exists(dummy_img_path):
                os.remove(dummy_img_path)

    def test_db_insert(self):
        test_data = {
            "store_name": "Test Supermarket",
            "total_amount": 1000,
            "tax_amount": 80,
            "items": [
                {
                    "japanese_name": "おにぎり",
                    "english_name": "Riceball",
                    "price": 150,
                    "category": "Food",
                    "cultural_context": "Japanese snack",
                    "confidence": 0.99
                }
            ],
            "savings_advice": "Save more money."
        }
        
        # Insert
        receipt_id = insert_receipt(test_data, "mock_image.png")
        self.assertGreater(receipt_id, 0)
        
        # Query and verify
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check receipts
        cursor.execute("SELECT store_name, total_amount, tax_amount FROM receipts WHERE id = ?", (receipt_id,))
        receipt_row = cursor.fetchone()
        self.assertEqual(receipt_row[0], "Test Supermarket")
        self.assertEqual(receipt_row[1], 1000)
        self.assertEqual(receipt_row[2], 80)
        
        # Check items
        cursor.execute("SELECT item_name, price, category FROM line_items WHERE receipt_id = ?", (receipt_id,))
        item_row = cursor.fetchone()
        self.assertEqual(item_row[0], "おにぎり")
        self.assertEqual(item_row[1], 150)
        self.assertEqual(item_row[2], "Food")
        
        conn.close()
 
    def test_simulation_parser(self):
        # Test the simulation fallback in ReceiptParser
        parser = ReceiptParser()
        result = parser._get_simulated_response()
        
        self.assertEqual(result["store_name"], "ファミリーマート 渋谷二丁目店 (FamilyMart)")
        self.assertEqual(result["total_amount"], 533)
        self.assertEqual(result["tax_amount"], 40)
        self.assertEqual(len(result["items"]), 4)
        self.assertEqual(result["items"][0]["note"], "FamilyMart's signature boneless fried chicken, highly popular among students as a quick hot snack.")

    def test_not_a_receipt_handling(self):
        parser = ReceiptParser()
        parser._call_groq_vision_with_retry = lambda *args, **kwargs: {"error": "not a receipt"}
        img = Image.new('RGB', (100, 100))
        with self.assertRaises(ValueError) as context:
            parser.parse_receipt_image(img, allow_simulation=False)
        self.assertIn("This image does not appear to be a receipt", str(context.exception))
        
if __name__ == "__main__":
    unittest.main()
