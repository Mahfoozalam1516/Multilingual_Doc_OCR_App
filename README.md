# Multilingual Document OCR Application

A Streamlit application for extracting text from documents in multiple languages using advanced image preprocessing techniques and OCR (Optical Character Recognition).

## Features

- Support for multiple languages
- Advanced image preprocessing
- Adjustable DPI settings
- Confidence score reporting
- Downloadable text output
- Real-time image processing preview
- Comprehensive scanning tips

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

#### Windows:

1. Download the Tesseract installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer
3. Make sure to check "Add to PATH" during installation
4. Select additional language packs during installation as needed

#### Linux:

```bash
sudo apt update
sudo apt install tesseract-ocr
# Install language packs (replace 'lang' with language code)
sudo apt install tesseract-ocr-lang
```

#### MacOS:

```bash
brew install tesseract
# Install language packs (replace 'lang' with language code)
brew install tesseract-lang
```

## Running the Application

### Local Development

```bash
streamlit run app.py
```

### Deploying to Streamlit Cloud

1. Create a Streamlit account at [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Deploy your app through the Streamlit dashboard

## Usage

1. Upload a document image (supported formats: JPG, JPEG, PNG, BMP, TIFF)
2. Select the language(s) present in your document
3. Adjust DPI settings if needed
4. Click "Extract Text" to process the document
5. View results and download extracted text

## Supported Languages

The application supports various languages including (but not limited to):

- English
- French
- German
- Spanish
- Italian
- Portuguese
- Russian
- Chinese (Simplified and Traditional)
- Japanese
- Korean
- Arabic
- Hindi
- Bengali
- Thai
- Vietnamese

Note: Language availability depends on installed Tesseract language packs.

## Project Structure

```
multilingual-ocr/
│
├── app.py                  # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore file
└── .streamlit/            # Streamlit configuration
    └── config.toml        # Streamlit config file
```

## Configuration

### Streamlit Config

Create a `.streamlit/config.toml` file:

```toml
[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
```

## Troubleshooting

1. **Tesseract Not Found Error**

   - Verify Tesseract is installed correctly
   - Check if Tesseract is added to PATH
   - Confirm installation path in the code matches your system

2. **Language Pack Issues**

   - Verify language packs are installed
   - Check language code usage
   - Install additional language packs as needed

3. **Image Processing Errors**
   - Ensure image is in supported format
   - Check image resolution and size
   - Verify image is not corrupted

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Tesseract OCR engine
- OpenCV for image processing
- Streamlit for the web interface
- PIL for image handling

## Support

For support, please open an issue in the repository or contact the maintainers.
