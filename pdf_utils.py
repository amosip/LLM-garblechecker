# pdf_utils.py

import fitz # PyMuPDF
import textwrap # Useful for sampling text
import os # Needed for os.path.join in extract_text_from_pdf error message

def extract_text_from_pdf(pdf_path):
    """
    Extracts text page by page from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The concatenated text from all pages, or None if an error occurs.
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        # Using os.path.basename for slightly cleaner error output
        print(f"Error extracting text from {os.path.basename(pdf_path)}: {e}")
        return None


def sample_text_from_string(text, num_samples=3, sample_length=1000):
    """
    Samples text snippets from a larger string.

    Args:
        text (str): The source text.
        num_samples (int): The number of snippets to sample.
        sample_length (int): The desired length of each snippet.

    Returns:
        list: A list of text snippets.
    """
    if not text or len(text.strip()) < sample_length:
        # If text is too short, just return the whole non-empty text as a single sample
        return [text.strip()] if text.strip() else []

    samples = []
    text_length = len(text)
    # Calculate step size to get roughly even distribution
    # Ensure step is at least sample_length to avoid overlapping too much on small texts
    step = max(sample_length, text_length // num_samples)

    for i in range(num_samples):
        # Calculate start index, ensuring it doesn't go beyond the text length
        start_index = min(i * step, text_length - sample_length)
        # Ensure end index doesn't go beyond text length
        end_index = start_index + sample_length
        # Add the snippet
        samples.append(text[start_index:end_index].strip())

    # Remove any empty samples that might result from stripping or small texts
    return [s for s in samples if s]