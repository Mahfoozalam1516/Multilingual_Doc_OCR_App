import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import spacy

# Set the Tesseract executable path (update this to your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\MAHFOOZ ALAM\anaconda3\Lib\site-packages\pytesseract\pytesseract.py"

# Load the English language model for spaCy
nlp = spacy.load('en_core_web_sm')

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Noise reduction
    denoised = cv2.fastNlMeansDenoising(binary)
    
    # Skew correction
    coords = np.column_stack(np.where(denoised > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = denoised.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def perform_ocr(image):
    text = pytesseract.image_to_string(image)
    return text

def post_process_text(text):
    doc = nlp(text)
    processed_text = ' '.join([token.text for token in doc if not token.is_space])
    return processed_text

def main():
    st.title("English Handwritten Text OCR App")
    st.write("Upload an image with handwritten English text for OCR")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Perform OCR"):
            with st.spinner("Processing..."):
                # Convert PIL Image to numpy array
                img_array = np.array(image)
                
                # Preprocess the image
                processed_img = preprocess_image(img_array)
                
                # Perform OCR
                text = perform_ocr(processed_img)
                
                # Post-process the text
                processed_text = post_process_text(text)
                
                st.subheader("OCR Result:")
                st.write(processed_text)

if __name__ == "__main__":
    main()