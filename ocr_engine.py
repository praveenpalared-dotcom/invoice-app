import os
import sys
import numpy as np
import pytesseract
from preprocessing import preprocess

_TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "tesseract")
pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD

_TESS_CONFIG = "--oem 3 --psm 6"


def extract_text(image_path: str, debug: bool = False) -> str:
    binary_img = preprocess(image_path, debug=debug)
    text = pytesseract.image_to_string(binary_img, config=_TESS_CONFIG)
    return text.strip()


def extract_text_with_confidence(image_path: str) -> tuple[str, float]:
    binary_img = preprocess(image_path)

    data = pytesseract.image_to_data(
        binary_img,
        config=_TESS_CONFIG,
        output_type=pytesseract.Output.DICT,
    )

    confidences = [int(c) for c in data["conf"] if int(c) != -1]
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

    text = pytesseract.image_to_string(binary_img, config=_TESS_CONFIG)
    return text.strip(), mean_conf


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_engine.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Running OCR on: {path}\n{'─'*50}")
    try:
        text, confidence = extract_text_with_confidence(path)
        print(text)
        print(f"\n{'─'*50}")
        print(f"✅  Mean OCR confidence: {confidence:.1f}%")
    except pytesseract.TesseractNotFoundError:
        print("❌  Tesseract not found. Install it and make sure it is on your PATH.")
        print("    Ubuntu: sudo apt-get install tesseract-ocr")
        print("    macOS : brew install tesseract")
        sys.exit(1)
