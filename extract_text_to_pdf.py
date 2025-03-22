# extract_text_to_pdf.py
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document
from fpdf import FPDF
import os
import traceback

# Set the Tesseract executable path explicitly
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image_pdf(file_path, output_dir="./example_data/classified"):
    """
    Extracts text from an image-based PDF using OCR and saves it to a new text-based PDF.
    Returns a list of Document objects and the path to the saved PDF.
    """
    try:
        # Check if input file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        # Verify Tesseract is accessible
        if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            raise FileNotFoundError(f"Tesseract executable not found at: {pytesseract.pytesseract.tesseract_cmd}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define output PDF path
        output_pdf_path = os.path.join(output_dir, "extracted_text.pdf")

        # Convert PDF to images
        print("Converting PDF to images...")
        images = convert_from_path(file_path, dpi=200)  # Adjust DPI for quality
        
        ocr_docs = []
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        for i, image in enumerate(images):
            print(f"Extracting text from page {i + 1}...")
            text = pytesseract.image_to_string(image, lang='eng')  # Specify language
            if text.strip():
                # Add to Document list for RAG
                ocr_docs.append(Document(
                    page_content=text,
                    metadata={"source": file_path, "page_label": str(i + 1)}
                ))
                # Add to PDF
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                # Encode text as UTF-8 to handle special characters
                pdf.multi_cell(0, 10, text.encode('latin-1', 'replace').decode('latin-1'))
            else:
                print(f"Warning: No text extracted from page {i + 1}")
        
        if not ocr_docs:
            print("No text extracted from the PDF.")
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, "No text extracted from PDF")
            ocr_docs = [Document(page_content="No text extracted from PDF", metadata={"source": file_path})]
        
        # Save the PDF
        pdf.output(output_pdf_path)
        print(f"Text extracted and saved to: {output_pdf_path}")
        return ocr_docs, output_pdf_path
    
    except Exception as e:
        print(f"Error during text extraction: {str(e)}")
        traceback.print_exc()
        return [Document(page_content=f"Error during extraction: {str(e)}", metadata={"source": file_path})], None

if __name__ == "__main__":
    # Example usage
    input_pdf = "./example_data/astral_experiment.pdf"
    docs, saved_pdf_path = extract_text_from_image_pdf(input_pdf)
    print(f"Extracted {len(docs)} pages. Saved PDF at: {saved_pdf_path}")