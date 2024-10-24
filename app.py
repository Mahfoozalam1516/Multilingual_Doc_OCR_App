import streamlit as st
import PIL.Image
import pytesseract
import numpy as np
import cv2
from io import BytesIO
import platform
import os
import re
from PIL import Image, ImageEnhance, ImageOps
import warnings
warnings.filterwarnings('ignore')

class DocumentPreprocessor:
    # [Previous DocumentPreprocessor code remains the same]
    @staticmethod
    def adjust_dpi(image, target_dpi=300):
        try:
            current_dpi = image.info.get('dpi', (72, 72))[0]
        except:
            current_dpi = 72
            
        if current_dpi < target_dpi:
            scale = target_dpi / current_dpi
            new_size = tuple(int(dim * scale) for dim in image.size)
            resized = image.resize(new_size, Image.Resampling.LANCZOS)
            resized.info['dpi'] = (target_dpi, target_dpi)
            return resized
        return image

    @staticmethod
    def binarization(image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            10
        )
        
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary, otsu
    
    @staticmethod
    def remove_noise(image):
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        kernel = np.ones((1, 1), np.uint8)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        return denoised

    @staticmethod
    def enhance_contrast(image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        return enhanced

    @staticmethod
    def deskew(image):
        coords = np.column_stack(np.where(image > 0))
        
        if len(coords) == 0:
            return image
            
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = 90 + angle
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, 
            M, 
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated

class OCRProcessor:
    def __init__(self):
        self.configure_tesseract()
        self.preprocessor = DocumentPreprocessor()
        self.supported_languages = self.get_supported_languages()
    
    def configure_tesseract(self):
        system = platform.system().lower()
        
        if system == 'windows':
            windows_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Users\User\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
            ]
            
            for path in windows_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    return True
            raise Exception("Tesseract not found. Please install it first.")
        
        try:
            pytesseract.get_tesseract_version()
        except:
            raise Exception("Tesseract not found. Please install it first.")

    def get_supported_languages(self):
        """Get list of installed Tesseract languages"""
        try:
            # Get list of supported languages
            languages = pytesseract.get_languages()
            
            # Create a dictionary of language codes and their full names
            language_names = {
                'eng': 'English',
                'fra': 'French',
                'deu': 'German',
                'spa': 'Spanish',
                'ita': 'Italian',
                'por': 'Portuguese',
                'rus': 'Russian',
                'chi_sim': 'Chinese (Simplified)',
                'chi_tra': 'Chinese (Traditional)',
                'jpn': 'Japanese',
                'kor': 'Korean',
                'ara': 'Arabic',
                'hin': 'Hindi',
                'ben': 'Bengali',
                'tha': 'Thai',
                'vie': 'Vietnamese'
                # Add more languages as needed
            }
            
            # Filter available languages
            available_languages = {}
            for lang in languages:
                if lang in language_names:
                    available_languages[lang] = language_names[lang]
            
            return available_languages
            
        except Exception as e:
            print(f"Error getting languages: {str(e)}")
            return {'eng': 'English'}  # Default to English if there's an error
    
    def preprocess_document(self, image, target_dpi):
        """Advanced document preprocessing pipeline with DPI adjustment"""
        if isinstance(image, Image.Image):
            image = self.preprocessor.adjust_dpi(image, target_dpi)
        
        image = np.array(image)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        enhanced = self.preprocessor.enhance_contrast(gray)
        denoised = self.preprocessor.remove_noise(enhanced)
        binary, otsu = self.preprocessor.binarization(denoised)
        deskewed = self.preprocessor.deskew(binary)
        
        return [binary, otsu, deskewed]
    
    def get_optimal_psm(self, image, lang_code):
        """Determine optimal PSM mode for given language"""
        psm_modes = [6, 3, 4]
        best_text = ""
        best_conf = 0
        best_psm = 6
        
        for psm in psm_modes:
            try:
                data = pytesseract.image_to_data(
                    image,
                    config=f'--psm {psm} --oem 3 -l {lang_code}',
                    output_type=pytesseract.Output.DICT
                )
                
                confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                if confidences:
                    avg_conf = sum(confidences) / len(confidences)
                    if avg_conf > best_conf:
                        best_conf = avg_conf
                        best_psm = psm
                        best_text = ' '.join([word for word in data['text'] if word.strip()])
            except:
                continue
        
        return best_psm
    
    def process_document(self, image, target_dpi=300, lang_codes='eng'):
        """Main document processing pipeline with multilingual support"""
        processed_images = self.preprocess_document(image, target_dpi)
        
        best_text = ""
        highest_confidence = 0
        best_processed_image = processed_images[0]
        
        for processed_image in processed_images:
            psm = self.get_optimal_psm(processed_image, lang_codes)
            
            # Configure for multiple languages
            custom_config = f'--oem 3 --psm {psm} -l {lang_codes}'
            
            try:
                data = pytesseract.image_to_data(
                    processed_image,
                    config=custom_config,
                    output_type=pytesseract.Output.DICT
                )
                
                confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    
                    if avg_confidence > highest_confidence:
                        highest_confidence = avg_confidence
                        text = ' '.join([word for word in data['text'] if word.strip()])
                        best_text = text
                        best_processed_image = processed_image
            except Exception as e:
                print(f"Error processing with language {lang_codes}: {str(e)}")
                continue
        
        return best_text, best_processed_image, highest_confidence

def main():
    st.set_page_config(page_title="Multilingual Document OCR App", layout="wide")
    
    st.title("Enhanced Multilingual Document OCR Application")
    st.write("Extract text from documents in multiple languages with advanced preprocessing")
    
    try:
        ocr_processor = OCRProcessor()
        
        uploaded_file = st.file_uploader("Upload a document image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            # Display original image
            image = Image.open(uploaded_file)
            col1.subheader("Original Document")
            col1.image(image, use_column_width=True)
            
            # Language selection
            available_languages = ocr_processor.supported_languages
            selected_languages = st.multiselect(
                "Select languages (in order of priority)",
                options=list(available_languages.keys()),
                default=['eng'],
                format_func=lambda x: f"{available_languages[x]} ({x})"
            )
            
            if not selected_languages:
                st.warning("Please select at least one language.")
                return
            
            # Create language string for Tesseract
            lang_string = '+'.join(selected_languages)
            
            # DPI settings
            target_dpi = st.slider(
                "Target DPI (higher values may improve accuracy but increase processing time)",
                min_value=72,
                max_value=600,
                value=300,
                step=72
            )
            
            current_dpi = image.info.get('dpi', (72, 72))[0]
            st.info(f"Current image DPI: {current_dpi:.0f}")
            
            # Process button
            if st.button('Extract Text', key='extract'):
                with st.spinner('Processing document... Please wait.'):
                    # Process document with specified languages and DPI
                    text, processed_image, confidence = ocr_processor.process_document(
                        image,
                        target_dpi=target_dpi,
                        lang_codes=lang_string
                    )
                    
                    # Display results
                    col2.subheader("Processed Document")
                    col2.image(processed_image, use_column_width=True)
                    
                    st.subheader("Extracted Text:")
                    if text.strip():
                        st.text_area("", text, height=200)
                        st.info(f"Confidence Score: {confidence:.2f}%")
                        
                        # Download button
                        st.download_button(
                            label="Download Text",
                            data=text.encode(),
                            file_name="extracted_text.txt",
                            mime="text/plain"
                        )
                    else:
                        st.warning("No text was detected. Please try adjusting the language selection, image quality, or DPI settings.")
            
            with st.expander("📋 Tips for Best Document Scanning"):
                st.markdown("""
                ### For Optimal Document Scanning:
                1. **Language Selection**
                   - Select the primary language of your document first
                   - For mixed-language documents, select all relevant languages
                   - Order languages by their prominence in the document
                
                2. **DPI Settings**
                   - 300 DPI is recommended for most documents
                   - Use higher DPI (400-600) for small text or poor quality images
                   - Lower DPI may be sufficient for clear, large text
                
                3. **Scanning Tips**
                   - Ensure document lies flat
                   - Avoid shadows and glare
                   - Use white background
                   - Consider using a scanner instead of camera for best results
                
                4. **Document Preparation**
                   - Clean, unwrinkled paper
                   - Clear text on white background
                   - Proper lighting
                   - Minimize background patterns
                
                5. **Language-Specific Tips**
                   - For right-to-left languages (Arabic, Hebrew), ensure proper orientation
                   - For Asian languages, higher DPI might be needed
                   - For mixed scripts, select all relevant language packs
                """)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        if "Tesseract not found" in str(e):
            st.markdown("""
            ### Installation Instructions:
            
            #### Windows:
            1. Download Tesseract installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
            2. Run installer (select "Add to PATH" during installation)
            3. Download additional language packs during installation
            
            #### Linux:
            ```bash
            sudo apt update
            sudo apt install tesseract-ocr
            # Install language packs (replace lang with desired language code)
            sudo apt install tesseract-ocr-lang
            ```
            
            #### MacOS:
            ```bash
            brew install tesseract
            # Install language packs (replace lang with desired language code)
            brew install tesseract-lang
            ```
            """)

if __name__ == '__main__':
    main()