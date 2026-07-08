import os
import sys
import base64
import requests
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

def transcribe():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("GROQ_API_KEY is empty! Please check your env setup.")
        sys.exit(1)

    image_path = None
    images = sorted([f for f in os.listdir("receipts") if f.startswith("Test1_") and f.endswith(".jpg")])
    if images:
        image_path = os.path.join("receipts", images[-1])
        
    if not image_path:
        print("No Yakiniku image found in receipts folder!")
        sys.exit(1)

    print(f"Reading Yakiniku receipt: {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Transcribe every single line of text on this receipt exactly, including all prices, totals, subtotals, tax details, payment methods, credit card details, cash received, change returned, etc. If it has Japanese characters, transcribe them exactly. Do not skip any line of text."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_string}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1024
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
    if response.status_code == 200:
        print("\n--- Transcription ---")
        print(response.json()["choices"][0]["message"]["content"])
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    transcribe()
