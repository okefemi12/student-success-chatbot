import io
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document
from pptx import Presentation
from openpyxl import load_workbook
from PIL import Image
import pytesseract

def extract_text_from_image(image_url):
    try:
        # 1. Download the image bytes from Cloudinary
        response = requests.get(image_url)
        response.raise_for_status() # Check for download errors
        
        # 2. Convert bytes to an Image object
        img = Image.open(BytesIO(response.content))
        
        # 3. Extract text
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""
    
# Extract text from PDF URL without saving to disk
def extract_text_from_pdf(file_url):
    try:
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
        
        # Read PDF directly from bytes in memory
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        return text.strip()
        
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""
    
# Extract text from DOCX URL without saving to disk
def extract_text_from_docx(file_url):
    try:
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
        
        # Read DOCX directly from bytes in memory
        docx_file = io.BytesIO(response.content)
        doc = Document(docx_file)
        
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
        return text.strip()
        
    except Exception as e:
        print(f"DOCX extraction error: {e}")
        return ""

# Extract text from PPTX URL without saving to disk 
def extract_text_from_pptx(file_url):
    try:
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
        
        # Read PPTX directly from bytes in memory
        pptx_file = io.BytesIO(response.content)
        prs = Presentation(pptx_file)
        
        text = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        
        return text.strip()
        
    except Exception as e:
        print(f"PPTX extraction error: {e}")
        return ""
    
def extract_text_from_excel(file_url):
    try:
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
        
        # Read Excel directly from bytes in memory
        excel_file = io.BytesIO(response.content)
        wb = load_workbook(excel_file)
        
        text = ""
        for sheet in wb.worksheets:
            for row in sheet.iter_rows(values_only=True):
                row_text = " ".join(str(cell) for cell in row if cell is not None)
                if row_text.strip():
                    text += row_text + "\n"
        
        return text.strip()
        
    except Exception as e:
        print(f"Excel extraction error: {e}")
        return ""