import cv2
import numpy as np
from pathlib import Path


def load_image(image_path: str) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_for_ocr(gray: np.ndarray, target_dpi_scale: float = 2.0) -> np.ndarray:
    h, w = gray.shape
    return cv2.resize(gray, (int(w * target_dpi_scale), int(h * target_dpi_scale)),
                      interpolation=cv2.INTER_CUBIC)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def deskew(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 50:
        return gray

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 0.5:
        return gray

    h, w = gray.shape
    centre = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )


def sharpen(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


def preprocess(image_path: str, debug: bool = False) -> np.ndarray:
    img = load_image(image_path)
    gray = to_grayscale(img)
    gray = resize_for_ocr(gray)
    gray = denoise(gray)
    gray = sharpen(gray)
    gray = deskew(gray)
    binary = binarize(gray)

    if debug:
        stem = Path(image_path).stem
        cv2.imwrite(f"debug_{stem}_gray.png", gray)
        cv2.imwrite(f"debug_{stem}_binary.png", binary)
        print(f"[preprocessing] Debug images saved for {stem}")

    return binary


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <image_path>")
        sys.exit(1)

    result = preprocess(sys.argv[1], debug=True)
    print(f"✅  Preprocessed image shape: {result.shape}  dtype: {result.dtype}")
