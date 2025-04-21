# This script checks the quality of text extracted from PDF files in a specified directory.

import fitz # PyMuPDF
import string
import os

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

def analyze_text_quality(text, min_line_chars=5, short_line_ratio_threshold=0.3, alphanumeric_ratio_threshold=0.6, printable_ratio_threshold=0.8):
    """
    Analyzes the quality of the extracted text for signs of garbling.

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
        dict: A dictionary containing quality metrics and a boolean flag 'is_garbled'.
    """
    if not text:
        return {
            'total_chars': 0,
            'total_lines': 0,
            'alphanumeric_ratio': 0.0,
            'printable_ratio': 0.0,
            'short_line_ratio': 0.0,
            'is_garbled': True, # Flag as garbled if no text is found
            'reason': 'No text extracted'
        }

    lines = text.strip().split('\n')
    total_lines = len(lines)
    total_chars = len(text)

    if total_chars == 0: # Should be caught by the first check, but defensive coding
         return {
            'total_chars': 0,
            'total_lines': total_lines,
            'alphanumeric_ratio': 0.0,
            'printable_ratio': 0.0,
            'short_line_ratio': 0.0,
            'is_garbled': True,
             'reason': 'Text is empty after stripping'
        }

    # Calculate alphanumeric ratio
    alphanumeric_chars = sum(c.isalnum() for c in text)
    alphanumeric_ratio = alphanumeric_chars / total_chars if total_chars > 0 else 0

    # Calculate printable ASCII ratio
    # We use string.printable which includes digits, ascii_letters, punctuation, and whitespace
    printable_chars = sum(c in string.printable for c in text)
    printable_ratio = printable_chars / total_chars if total_chars > 0 else 0

    # Calculate ratio of short lines
    short_lines_count = 0
    if total_lines > 0:
        for line in lines:
            if len(line.strip()) < min_line_chars and len(line.strip()) > 0: # Ignore completely empty lines
                 short_lines_count += 1
        short_line_ratio = short_lines_count / total_lines
    else:
        short_line_ratio = 0.0

    # Determine if garbled based on thresholds
    is_garbled = False
    reason = []

    if alphanumeric_ratio < alphanumeric_ratio_threshold:
        is_garbled = True
        reason.append(f"Low alphanumeric ratio ({alphanumeric_ratio:.2f})")
    if printable_ratio < printable_ratio_threshold:
        is_garbled = True
        reason.append(f"Low printable character ratio ({printable_ratio:.2f})")
    if short_line_ratio > short_line_ratio_threshold:
        is_garbled = True
        reason.append(f"High short line ratio ({short_line_ratio:.2f})")

    if not is_garbled:
        reason.append("Looks OK")


    return {
        'total_chars': total_chars,
        'total_lines': total_lines,
        'alphanumeric_ratio': alphanumeric_ratio,
        'printable_ratio': printable_ratio,
        'short_line_ratio': short_line_ratio,
        'is_garbled': is_garbled,
        'reason': ', '.join(reason)
    }

def check_pdf_directory(directory_path, **analysis_params):
    """
    Checks all PDF files in a given directory.

    Args:
        directory_path (str): The path to the directory containing PDF files.
        **analysis_params: Keyword arguments to pass to analyze_text_quality
                         (e.g., min_line_chars=10, short_line_ratio_threshold=0.4).

    Returns:
        list: A list of dictionaries, each containing the file path and its analysis results.
    """
    results = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            print(f"Checking: {pdf_path}")
            extracted_text = extract_text_from_pdf(pdf_path)
            if extracted_text is not None:
                analysis_result = analyze_text_quality(extracted_text, **analysis_params)
                results.append({'file': filename, 'result': analysis_result})
            else:
                 results.append({'file': filename, 'result': {'is_garbled': True, 'reason': 'Extraction failed'}})

    return results

if __name__ == "__main__":
    # --- Configuration ---
    # Replace with the path to your directory containing PDF files
    pdf_directory = "./your_pdf_directory" # <--- IMPORTANT: Change this path

    # --- Analysis Thresholds (Adjust as needed) ---
    # Minimum non-whitespace characters for a line to be considered non-short
    min_line_chars_threshold = 10

    # If the ratio of lines shorter than min_line_chars exceeds this, flag it
    short_line_threshold_ratio = 0.25 # e.g., if > 25% of lines are very short

    # If the ratio of alphanumeric chars is below this, flag it
    alphanumeric_threshold_ratio = 0.5 # e.g., if < 50% of chars are letters/numbers

    # If the ratio of standard printable ASCII chars is below this, flag it
    printable_threshold_ratio = 0.7 # e.g., if < 70% of chars are printable ASCII

    # ---------------------

    if not os.path.isdir(pdf_directory):
        print(f"Error: Directory not found at {pdf_directory}")
    else:
        print(f"Starting text quality check for PDFs in: {pdf_directory}\n")
        analysis_results = check_pdf_directory(
            pdf_directory,
            min_line_chars=min_line_chars_threshold,
            short_line_ratio_threshold=short_line_threshold_ratio,
            alphanumeric_ratio_threshold=alphanumeric_threshold_ratio,
            printable_ratio_threshold=printable_threshold_ratio
        )

        print("\n--- Summary ---")
        garbled_files = []
        ok_files = []

        for item in analysis_results:
            file = item['file']
            result = item['result']
            print(f"File: {file}")
            print(f"  Total Chars: {result['total_chars']}")
            print(f"  Total Lines: {result['total_lines']}")
            print(f"  Alphanumeric Ratio: {result['alphanumeric_ratio']:.2f}")
            print(f"  Printable Ratio:    {result['printable_ratio']:.2f}")
            print(f"  Short Line Ratio:   {result['short_line_ratio']:.2f}")
            print(f"  Is Garbled: {result['is_garbled']}")
            print(f"  Reason: {result['reason']}\n")

            if result['is_garbled']:
                garbled_files.append(file)
            else:
                ok_files.append(file)

        print("\n--- Potential Garbled Files ---")
        if garbled_files:
            for f in garbled_files:
                print(f"- {f}")
        else:
            print("None")

        print("\n--- Files Looking OK ---")
        if ok_files:
             for f in ok_files:
                print(f"- {f}")
        else:
            print("None")