# Add these two lines at the very top to load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

import fitz # PyMuPDF
import string
import os
import textwrap # Useful for sampling text
import json # Needed to parse JSON response from LLM
from groq import Groq # Import the Groq client

# --- Groq Configuration ---
# It's recommended to set GROQ_API_KEY environment variable in a .env file
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# Using the model you specified
GROQ_MODEL = "deepseek-r1-distill-llama-70b"
LLM_TEMP = 0.1 # Lower temperature for more consistent outputs

def get_groq_client():
    """Initializes and returns the Groq client."""
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY environment variable not set.")
        print("Please ensure you have a .env file in the project root with GROQ_API_KEY='your_key_here'")
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        # Optional: Test client connection
        # client.models.list()
        return client
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        print("Please double-check your GROQ_API_KEY and network connection.")
        return None

# Initialize client globally (will be None if key is missing)
groq_client = get_groq_client()


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
        print(f"Error extracting text from {pdf_path}: {e}")
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
        # If text is too short, just return the whole non-empty text
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


# --- Statistical Analysis Function (Placed BEFORE comprehensive_text_analysis) ---
def analyze_text_quality(text, min_line_chars=5, short_line_ratio_threshold=0.3, alphanumeric_ratio_threshold=0.6, printable_ratio_threshold=0.8):
    """
    Analyzes the quality of the extracted text for signs of garbling
    using statistical metrics.

    Args:
        text (str): The text extracted from the PDF.
        min_line_chars (int): Minimum number of non-whitespace characters for a line to be considered non-short.
        short_line_ratio_threshold (float): If the ratio of lines shorter than min_line_chars
                                            exceeds this, it's a potential issue.
        alphanumeric_ratio_threshold (float): If the ratio of alphanumeric characters
                                             to total characters is below this, it's a potential issue.
        printable_ratio_threshold (float): If the ratio of standard printable ASCII characters
                                           to total characters is below this, it's a potential issue.

    Returns:
        dict: A dictionary containing quality metrics and a boolean flag 'is_garbled'
              based on statistical checks alone.
    """
    if not text:
        return {
            'total_chars': 0,
            'total_lines': 0,
            'alphanumeric_ratio': 0.0,
            'printable_ratio': 0.0,
            'short_line_ratio': 0.0,
            'is_garbled_statistical': True, # Flag specifically for statistical garble
            'statistical_reason': 'No text extracted'
        }

    lines = text.strip().split('\n')
    total_lines = len(lines)
    total_chars = len(text)

    if total_chars == 0:
         return {
            'total_chars': 0,
            'total_lines': total_lines,
            'alphanumeric_ratio': 0.0,
            'printable_ratio': 0.0,
            'short_line_ratio': 0.0,
            'is_garbled_statistical': True,
            'statistical_reason': 'Text is empty after stripping'
        }

    # Calculate alphanumeric ratio
    alphanumeric_chars = sum(c.isalnum() for c in text)
    alphanumeric_ratio = alphanumeric_chars / total_chars if total_chars > 0 else 0

    # Calculate printable ASCII ratio
    printable_chars = sum(c in string.printable for c in text)
    printable_ratio = printable_chars / total_chars if total_chars > 0 else 0

    # Calculate ratio of short lines
    short_lines_count = 0
    if total_lines > 0:
        for line in lines:
            # Consider lines with only whitespace or very few chars as potentially problematic
            if len(line.strip()) < min_line_chars and len(line.strip()) > 0:
                 short_lines_count += 1
        short_line_ratio = short_lines_count / total_lines
    else:
        short_line_ratio = 0.0

    # Determine if garbled based on statistical thresholds
    is_garbled_statistical = False
    statistical_reason_list = []

    if alphanumeric_ratio < alphanumeric_ratio_threshold:
        is_garbled_statistical = True
        statistical_reason_list.append(f"Stat: Low alphanumeric ratio ({alphanumeric_ratio:.2f}) < {alphanumeric_ratio_threshold}")
    if printable_ratio < printable_ratio_threshold:
        is_garbled_statistical = True
        statistical_reason_list.append(f"Stat: Low printable character ratio ({printable_ratio:.2f}) < {printable_ratio_threshold}")
    if short_line_ratio > short_line_ratio_threshold:
        is_garbled_statistical = True
        statistical_reason_list.append(f"Stat: High short line ratio ({short_line_ratio:.2f}) > {short_line_ratio_threshold}")