#new refactored pdf_check.py script, borken out into functions and classes for better readability and maintainability


from pdf_utils import extract_text_from_pdf, sample_text_from_string
from dotenv import load_dotenv
load_dotenv()
import csv
import string
from datetime import datetime
import fitz # PyMuPDF
import string
# os is still needed for file path operations and env vars
import os # Needed for os.path.join in extract_text_from_pdf error message
import json # Needed to parse JSON response from LLM
from groq import Groq # Import the Groq client
from tabulate import tabulate # Import tabulate for table formatting


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


# --- LLM Check Method 1: Fluency and Coherence ---
def check_fluency_with_llm(text_snippet, client):
    """
    Checks the fluency and coherence of a text snippet using an LLM.

    Args:
        text_snippet (str): The text snippet to check.
        client: The initialized Groq client.

    Returns:
        dict: Result {'score': float, 'reason': str} or {'error': str}.
    """
    if not client:
         return {'score': 0.0, 'reason': 'LLM client not initialized'}
    if not text_snippet or not text_snippet.strip():
        return {'score': 0.0, 'reason': 'No text snippet provided'}

    prompt = f"""
    Analyze the following text snippet for its fluency, coherence, and whether it appears to be standard human-readable text (like from a document or book) versus being garbled, corrupted, or non-sensical data.

    Provide a score between 0.0 (completely garbled/unreadable) and 1.0 (perfectly fluent and coherent). Also, provide a brief reason for your score.

    Format your output as a JSON object with keys "score" (float) and "reason" (string).

    Text Snippet:
    ---
    {text_snippet[:2000]} # Limit snippet length for prompt to manage token usage
    ---
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a text quality assessment AI. You evaluate if text is fluent and coherent. Provide JSON output only."},
                {"role": "user", "content": prompt},
            ],
            model=GROQ_MODEL,
            temperature=LLM_TEMP,
            response_format={"type": "json_object"} # Request JSON output
        )
        response_content = chat_completion.choices[0].message.content
        # Attempt to parse the JSON response
        try:
            llm_result = json.loads(response_content)
            if 'score' in llm_result and 'reason' in llm_result:
                 # Ensure score is a float and is within the 0-1 range
                score = float(llm_result['score'])
                llm_result['score'] = max(0.0, min(1.0, score)) # Clamp score between 0 and 1
                llm_result['reason'] = f"LLM Fluency: {llm_result['reason']}"
                return llm_result
            else:
                 return {'score': 0.0, 'reason': f'LLM Fluency: Returned invalid JSON format: {response_content}'}
        except json.JSONDecodeError:
            return {'score': 0.0, 'reason': f'LLM Fluency: Returned non-JSON response: {response_content}'}

    except Exception as e:
        return {'score': 0.0, 'reason': f"LLM Fluency Error: {e}"}


# --- LLM Check Method 2: Attempted Task Performance (Summarization) ---
def check_summarization_with_llm(text_snippet, client):
    """
    Attempts to summarize a text snippet using an LLM as a check for usability.
    Success indicates usable text, failure indicates potential garbling.

    Args:
        text_snippet (str): The text snippet to attempt to summarize.
        client: The initialized Groq client.

    Returns:
        dict: Result {'success': bool, 'summary': str, 'reason': str} or {'error': str}.
    """
    if not client:
        return {'success': False, 'summary': '', 'reason': 'LLM client not initialized'}
    if not text_snippet or not text_snippet.strip():
         return {'success': False, 'summary': '', 'reason': 'No text snippet provided'}

    prompt = f"""
    Attempt to provide a very brief summary (1-2 sentences) of the following text snippet. If the text is garbled, nonsensical, or cannot be summarized, please state clearly and concisely *that* it cannot be summarized and explain *why* in the summary field, rather than trying to summarize.

    Text Snippet:
    ---
    {text_snippet[:2000]} # Limit snippet length for prompt
    ---
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text or explains why it cannot be summarized."},
                {"role": "user", "content": prompt},
            ],
            model=GROQ_MODEL,
            temperature=LLM_TEMP,
        )
        response_content = chat_completion.choices[0].message.content

        # Simple check for indicators of failure in the LLM's response
        # We look for phrases that indicate it couldn't summarize successfully
        failure_indicators = [
            "cannot be summarized",
            "unable to summarize",
            "text is garbled",
            "text is nonsensical",
            "difficult to understand",
            "appears to be random characters"
            # Add more indicators if you observe the LLM using other phrases for failure
        ]
        # Check if the response content contains any of the failure indicators (case-insensitive)
        is_successful = not any(indicator in response_content.lower() for indicator in failure_indicators)

        reason = 'LLM Summarization: Summarized successfully' if is_successful else 'LLM Summarization: Failed to summarize'

        return {'success': is_successful, 'summary': response_content, 'reason': reason}

    except Exception as e:
         return {'success': False, 'summary': '', 'reason': f"LLM Summarization Error: {e}"}


# --- Combined Analysis Function ---
def comprehensive_text_analysis(text, client, num_llm_samples=2, **statistical_params):
    """
    Performs both statistical and LLM-based text quality analysis.

    Args:
        text (str): The full text extracted from the PDF.
        client: The initialized Groq client (can be None).
        num_llm_samples (int): Number of text snippets to sample for LLM checks.
        **statistical_params: Keyword arguments for the statistical analysis.

    Returns:
        dict: Combined analysis results.
    """
    # Perform statistical analysis first
    statistical_results = analyze_text_quality(text, **statistical_params)

    # Initialize combined results with statistical results
    combined_results = statistical_results.copy()
    combined_results['llm_status'] = "Skipped (LLM client not initialized or no text)"
    # Initialize overall garbled and reason based on statistical results initially
    combined_results['is_garbled'] = statistical_results['is_garbled_statistical']
    combined_results['reason'] = statistical_results['statistical_reason']


    if client and text and text.strip():
        # Use the sampling function; it handles cases with little text
        text_snippets = sample_text_from_string(text, num_samples=num_llm_samples, sample_length=1000) # Use sample_text_from_string with potentially larger length for LLM

        if not text_snippets:
             combined_results['llm_status'] = "Skipped (No usable text snippets for LLM)"
        else:
            combined_results['llm_samples'] = text_snippets
            combined_results['llm_status'] = "Performed"


            # Run Fluency Check on samples
            fluency_checks = []
            print(f"  Running {len(text_snippets)} LLM Fluency Check(s)...")
            for i, snippet in enumerate(text_snippets):
                 # print(f"    Sample {i+1}/{len(text_snippets)}...") # Optional: print for each sample
                 fluency_checks.append(check_fluency_with_llm(snippet, client))
            combined_results['fluency_checks'] = fluency_checks

            # Run Summarization Check on samples
            summarization_checks = []
            print(f"  Running {len(text_snippets)} LLM Summarization Check(s)...")
            for i, snippet in enumerate(text_snippets):
                 # print(f"    Sample {i+1}/{len(text_snippets)}...") # Optional: print for each sample
                 summarization_checks.append(check_summarization_with_llm(snippet, client))
            combined_results['summarization_checks'] = summarization_checks

            # Aggregate LLM results to influence the overall 'is_garbled' flag
            # Simple aggregation: if any sample strongly indicates garbling by LLM
            # Filter out errors from calculation
            valid_fluency_scores = [c['score'] for c in fluency_checks if 'score' in c and c.get('score') is not None]
            avg_fluency_score = sum(valid_fluency_scores) / len(valid_fluency_scores) if valid_fluency_scores else 0.0 # Default to 0 if no valid scores

            valid_summarization_results = [c for c in summarization_checks if 'success' in c]
            summarization_failures = sum(not c['success'] for c in valid_summarization_results)
            total_llm_samples_run = len(text_snippets) # Base failure count on actual samples run


            # Define thresholds for LLM flagging (can adjust these based on testing)
            LLM_FLUENCY_THRESHOLD = 0.6 # If average fluency score is below this, flag
            # If more than a certain ratio or count of samples fail summarization, flag
            LLM_SUMMARIZATION_FAILURE_THRESHOLD_RATIO = 0.5 # e.g., > 50% samples fail

            llm_reasons = []

            # Check if LLM samples were actually run before applying LLM thresholds
            if total_llm_samples_run > 0:
                if avg_fluency_score < LLM_FLUENCY_THRESHOLD:
                    combined_results['is_garbled'] = True
                    llm_reasons.append(f"LLM: Avg Fluency ({avg_fluency_score:.2f}) < {LLM_FLUENCY_THRESHOLD}")

                if summarization_failures / total_llm_samples_run > LLM_SUMMARIZATION_FAILURE_THRESHOLD_RATIO:
                    combined_results['is_garbled'] = True
                    llm_reasons.append(f"LLM: Summarization Failed ({summarization_failures}/{total_llm_samples_run} > {LLM_SUMMARIZATION_FAILURE_THRESHOLD_RATIO*100:.0f}%)")

            # Update the combined reason string
            if llm_reasons:
                combined_results['reason'] = statistical_results['statistical_reason'] + ", " + ", ".join(llm_reasons)


    return combined_results


# --- Main function to check PDFs in a directory ---
def check_pdf_directory(directory_path, client, num_llm_samples=2, **analysis_params):
    """
    Checks all PDF files in a given directory using comprehensive analysis.

    Args:
        directory_path (str): The path to the directory containing PDF files.
        client: The initialized Groq client (can be None).
        num_llm_samples (int): Number of text snippets to sample for LLM checks per PDF.
        **analysis_params: Keyword arguments to pass to comprehensive_text_analysis.

    Returns:
        list: A list of dictionaries, each containing the file path and its analysis results.
    """
    results = []
    # Use a consistent order for processing files
    for filename in sorted(os.listdir(directory_path)):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            print(f"--- Checking: {filename} ---") # Use filename for cleaner output
            extracted_text = extract_text_from_pdf(pdf_path)

            # Pass the Groq client and num_llm_samples to the comprehensive analysis
            analysis_result = comprehensive_text_analysis(
                extracted_text,
                client, # Pass the client here
                num_llm_samples=num_llm_samples, # Pass the number of samples
                **analysis_params
            )
            results.append({'file': filename, 'result': analysis_result})
            print("-" * (len(filename) + 12) + "\n") # Separator line


    return results


# --- Script Entry Point ---
if __name__ == "__main__":
    # Ensure the Groq client was initialized successfully before starting main process
    # This check is done inside get_groq_client, but we print a status here
    if groq_client is None and os.environ.get("GROQ_API_KEY"):
        print("Groq client failed to initialize. LLM checks will be skipped for all files.")
    elif groq_client is None:
        print("GROQ_API_KEY not set. LLM checks will be skipped for all files.")
    else:
        print("Groq client initialized successfully. LLM checks will be performed for files with sufficient text.")


    # --- Configuration ---
    # Replace with the path to your directory containing PDF files
    pdf_directory = "./pdfs_to_check" # <--- IMPORTANT: Change this path

    # --- Analysis Thresholds (Adjust as needed) ---
    # Statistical Thresholds
    min_line_chars_threshold = 10
    short_line_threshold_ratio = 0.25
    alphanumeric_threshold_ratio = 0.5
    printable_threshold_ratio = 0.7

    # LLM Check Parameters
    num_llm_samples_per_pdf = 3 # How many snippets to check with LLM per PDF
    # Note: LLM sample length is defined within sample_text_from_string (currently 1000 chars)

    # LLM Thresholds (within comprehensive_text_analysis function - adjust there)
    # LLM_FLUENCY_THRESHOLD = 0.6
    # LLM_SUMMARIZATION_FAILURE_THRESHOLD_RATIO = 0.5


    # ---------------------

    if not os.path.isdir(pdf_directory):
        print(f"Error: Directory not found at {pdf_directory}")
    else:
        print(f"Starting text quality check for PDFs in: {pdf_directory}\n")
        analysis_results = check_pdf_directory(
            pdf_directory,
            groq_client, # Pass the Groq client
            num_llm_samples=num_llm_samples_per_pdf, # Pass LLM sample count
            min_line_chars=min_line_chars_threshold,
            short_line_ratio_threshold=short_line_threshold_ratio,
            alphanumeric_ratio_threshold=alphanumeric_threshold_ratio,
            printable_ratio_threshold=printable_threshold_ratio
        )

              # --- Prepare Data for Table ---
        table_data = []
        headers = [
            "File Name",
            "Overall Garbled",
            "Stat Reason",
            "Alpha Ratio",
            "Print Ratio",
            "Short Line Ratio",
            "LLM Status",
            "Avg Fluency",
            "Summary Failures",
            "Combined Reason"
        ]

        # Sort results by overall status (Garbled first), then filename
        sorted_results = sorted(analysis_results, key=lambda x: (not x['result'].get('is_garbled', True), x['file']))


        for item in sorted_results:
            file = item['file']
            result = item['result']

            # Extract statistical metrics
            alpha_ratio = result.get('alphanumeric_ratio', 0.0)
            print_ratio = result.get('printable_ratio', 0.0)
            short_line_ratio = result.get('short_line_ratio', 0.0)
            stat_reason = result.get('statistical_reason', 'N/A')

            # Extract LLM metrics
            llm_status = result.get('llm_status', 'N/A')
            avg_fluency = 'N/A'
            summary_failures_str = 'N/A' # Will show count/total
            combined_reason = result.get('reason', 'Unknown')
            overall_garbled = result.get('is_garbled', True)


            if llm_status == "Performed":
                 fluency_checks = result.get('fluency_checks', [])
                 valid_fluency_scores = [c['score'] for c in fluency_checks if 'score' in c and c.get('score') is not None]
                 if valid_fluency_scores:
                    avg_fluency = f"{sum(valid_fluency_scores) / len(valid_fluency_scores):.2f}"
                 else:
                     avg_fluency = 'Calc Err' # Should not happen with current logic, but good to handle

                 summarization_checks = result.get('summarization_checks', [])
                 valid_summarization_results = [c for c in summarization_checks if 'success' in c]
                 if valid_summarization_results:
                    summarization_failures = sum(not c['success'] for c in valid_summarization_results)
                    total_llm_checks_run = len(valid_summarization_results)
                    summary_failures_str = f"{summarization_failures}/{total_llm_checks_run}"
                 else:
                     summary_failures_str = 'Calc Err' # Should not happen


            # Append data row - Ensure data types are simple strings/numbers for CSV
            table_data.append([
                file,
                "YES" if overall_garbled else "NO",
                stat_reason,
                f"{alpha_ratio:.2f}",
                f"{print_ratio:.2f}",
                f"{short_line_ratio:.2f}",
                llm_status,
                avg_fluency,
                summary_failures_str,
                combined_reason
            ])

        # --- Define CSV File Path ---
        # Create a timestamped filename to avoid overwriting previous results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"pdf_analysis_results_{timestamp}.csv"
        # You can change the directory where the CSV is saved if needed
        # csv_filepath = os.path.join("./results", csv_filename)


        # --- Write Data to CSV ---
        try:
            with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)

                # Write the headers
                writer.writerow(headers)

                # Write the data rows
                writer.writerows(table_data)

            print(f"\nAnalysis results successfully exported to {csv_filename}")

        except IOError as e:
            print(f"\nError writing CSV file {csv_filename}: {e}")


        # --- Print the Table (Optional - you can remove this if you only want CSV) ---
        # You still need 'from tabulate import tabulate' at the top if you keep this
        try:
            from tabulate import tabulate
            print("\n" + "="*60)
            print("--- TEXT QUALITY ANALYSIS SUMMARY (Console View) ---")
            print("="*60 + "\n")
            print(tabulate(table_data, headers=headers, tablefmt="grid", maxcolwidths=[None, None, 30, None, None, None, None, None, None, 50]))
            print("="*60 + "\n") # Closing line for console table
        except ImportError:
             print("\nTabulate library not found. Skipping console table output.")


        # Print the final lists below the table for easy copying
        garbled_files = [item['file'] for item in sorted_results if item['result'].get('is_garbled', True)]
        ok_files = [item['file'] for item in sorted_results if not item['result'].get('is_garbled', True)]

        print("\n" + "="*30 + "\n--- POTENTIAL GARBLED FILES (Overall) ---")
        if garbled_files:
            for f in garbled_files:
                print(f"- {f}")
        else:
            print("None")

        print("\n" + "="*30 + "\n--- FILES LOOKING OK (Overall) ---")
        if ok_files:
             for f in ok_files:
                print(f"- {f}")
        else:
            print("None")
        print("="*30 + "\n")