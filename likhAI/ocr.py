import pytesseract
from PIL import Image
import cv2
import numpy as np

# Add the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# def preprocess_image(image):
#     gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

#     _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     processed_image = cv2.medianBlur(threshold_image, 3)

#     return processed_image

# def extract_text(image_path):
#     try:
#         image = Image.open(image_path)
        
#         processed_image = preprocess_image(image)
        
#         text = pytesseract.image_to_string(processed_image)

#         return text
    
#     except Exception as e:
#         return f"Error processing image: {str(e)}"

def preprocess_image(image):
    # Convert the image to a numpy array
    image_np = np.array(image)

    # Check the number of channels in the image
    if len(image_np.shape) == 2:  # Grayscale image (already single channel)
        gray_image = image_np
    elif len(image_np.shape) == 3:  # Color image (RGB or RGBA)
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Unsupported image format")

    # Apply thresholding
    _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply median blur
    processed_image = cv2.medianBlur(threshold_image, 3)

    return processed_image

def extract_text(image):
    try:
        processed_image = preprocess_image(image)
        text = pytesseract.image_to_string(processed_image)
        return text
    except Exception as e:
        return f"Error processing image: {str(e)}"