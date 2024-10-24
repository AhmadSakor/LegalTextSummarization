import os
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import cv2
import numpy as np
import re
from multiprocessing import Pool, cpu_count
import argparse

def preprocess_image(image):
    """
    Preprocesses the image for better OCR results by:
    - Converting the image to grayscale.
    - Applying thresholding for better contrast.
    - Removing noise using Non-local Means Denoising.
    
    Args:
        image (PIL.Image): Input image to preprocess.
    
    Returns:
        PIL.Image: Preprocessed image ready for OCR.
    """
    # Convert PIL Image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding to binarize the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply fast Non-local Means Denoising to reduce noise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    return Image.fromarray(denoised)

def post_process_text(text):
    """
    Cleans the OCR-extracted text by:
    - Removing non-Arabic characters except numbers and some punctuation.
    - Removing extra whitespace for better readability.
    
    Args:
        text (str): Raw text extracted by OCR.
    
    Returns:
        str: Cleaned and formatted text.
    """
    arabic_pattern = re.compile(r'[^\u0600-\u06FF\s0-9.,!?()[\]{}:;\-\'"]+')
    cleaned_text = arabic_pattern.sub('', text)
    cleaned_text = ' '.join(cleaned_text.split())
    
    return cleaned_text

def extract_text_from_pdf(pdf_path, tesseract_cmd, tessdata_prefix):
    """
    Converts a PDF file to images and extracts text from each page using Tesseract OCR.
    
    Args:
        pdf_path (str): Path to the PDF file.
        tesseract_cmd (str): Path to the Tesseract command.
        tessdata_prefix (str): Path to the TESSDATA_PREFIX environment.
    
    Returns:
        str: Extracted and cleaned text from all PDF pages.
    """
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    os.environ['TESSDATA_PREFIX'] = tessdata_prefix

    images = convert_from_path(pdf_path)
    text = ""
    for i, image in enumerate(images):
        preprocessed_image = preprocess_image(image)
        raw_text = pytesseract.image_to_string(
            preprocessed_image, 
            lang='ara',
            config='--psm 6 --oem 1 -c preserve_interword_spaces=1'
        )
        cleaned_text = post_process_text(raw_text)
        text += f"--- Page {i+1} ---\n{cleaned_text}\n\n"
    
    return text

def process_single_pdf(args):
    pdf_path, output_folder, tesseract_cmd, tessdata_prefix = args
    filename = os.path.basename(pdf_path)
    
    try:
        extracted_text = extract_text_from_pdf(pdf_path, tesseract_cmd, tessdata_prefix)
        txt_filename = os.path.splitext(filename)[0] + '_ocr.txt'
        txt_path = os.path.join(output_folder, txt_filename)
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(extracted_text)
        return f"Successfully processed: {filename}"
    except Exception as e:
        return f"Error processing {filename}: {str(e)}"

def process_pdfs_parallel(folder_path, output_folder, num_processes, tesseract_cmd, tessdata_prefix):
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    os.makedirs(output_folder, exist_ok=True)
    
    with Pool(num_processes) as pool:
        results = pool.map(process_single_pdf, [(pdf, output_folder, tesseract_cmd, tessdata_prefix) for pdf in pdf_files])
    
    for result in results:
        print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process PDFs using Tesseract OCR.')
    parser.add_argument('--pdf_folder', type=str, required=True, help='Folder containing PDF files.')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save extracted text files.')
    parser.add_argument('--tesseract_cmd', type=str, default='/usr/bin/tesseract', help='Path to Tesseract command.')
    parser.add_argument('--tessdata_prefix', type=str, default='/usr/share/tesseract-ocr/4.00/tessdata/', help='Path to TESSDATA_PREFIX.')
    parser.add_argument('--num_processes', type=int, default=max(1, cpu_count() - 4), help='Number of processes for parallel processing.')

    args = parser.parse_args()

    process_pdfs_parallel(args.pdf_folder, args.output_folder, args.num_processes, args.tesseract_cmd, args.tessdata_prefix)
