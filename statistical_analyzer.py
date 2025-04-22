# statistical_analyzer.py

import string # Needed for string.printable

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
        statistical_reason_list.append(f"Stat: Low alpha ratio ({alphanumeric_ratio:.2f} < {alphanumeric_ratio_threshold})")
    if printable_ratio < printable_ratio_threshold:
        is_garbled_statistical = True
        statistical_reason_list.append(f"Stat: Low printable ratio ({printable_ratio:.2f} < {printable_ratio_threshold})")
    if short_line_ratio > short_line_ratio_threshold:
        is_garbled_statistical = True
        statistical_reason_list.append(f"Stat: High short line ratio ({short_line_ratio:.2f} > {short_line_ratio_threshold})")

    if not is_garbled_statistical:
        statistical_reason_list.append("Stat: Looks OK")


    return {
        'total_chars': total_chars,
        'total_lines': total_lines,
        'alphanumeric_ratio': alphanumeric_ratio,
        'printable_ratio': printable_ratio,
        'short_line_ratio': short_line_ratio,
        'is_garbled_statistical': is_garbled_statistical,
        'statistical_reason': ', '.join(statistical_reason_list)
    }