#new refactored pdf_check.py script, borken out into functions and classes for better readability and maintainability

# Imports for utilities
from pdf_utils import extract_text_from_pdf, sample_text_from_string
# Add this import for statistical analysis
from statistical_analyzer import analyze_text_quality

# Standard library imports
from dotenv import load_dotenv
import csv
# string is no longer needed in this file if only analyze_text_quality used it
# import string # Removed as it's likely used within statistical_analyzer now
from datetime import datetime
# fitz is no longer needed here if extract_text_from_pdf is imported
# import fitz # PyMuPDF
import os # Needed for file path operations and env vars
import json # Needed to parse JSON response from LLM

# Third-party imports
from groq import Groq # Import the Groq client
from tabulate import tabulate # Import tabulate for table formatting

# Load environment variables
load_dotenv()


# --- Groq Configuration ---
# It's recommended to set GROQ_API_KEY environment variable in a .env file
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# Using the model you specified
GROQ_MODEL = "deepseek-r1-distill-llama-70b" # Or another suitable model like mixtral-8x7b-32768
LLM_TEMP = 0.1 # Lower temperature for more consistent outputs

def get_groq_client():
    """Initializes and returns the Groq client."""
    if not GROQ_API_KEY:
        print("Warning: GROQ_API_KEY environment variable not set.")
        print("LLM checks will be skipped. Ensure you have a .env file with GROQ_API_KEY='your_key_here'")
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        # Optional: Test client connection (uncomment if needed)
        # client.models.list()
        # print("Groq client initialized successfully.") # Optional confirmation
        return client
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        print("Please double-check your GROQ_API_KEY and network connection.")
        return None

# Initialize client globally (will be None if key is missing)
groq_client = get_groq_client()


# --- PDF Utility Functions (Now Imported) ---
# extract_text_from_pdf is now imported from pdf_utils
# sample_text_from_string is now imported from pdf_utils


# --- Statistical Analysis Function (Now Imported) ---
# analyze_text_quality is now imported from statistical_analyzer


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

    Format your output as a JSON object with keys "score" (float) and "reason" (string). Respond ONLY with the JSON object.

    Text Snippet:
    ---
    {text_snippet[:2000]} # Limit snippet length for prompt to manage token usage
    ---
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a text quality assessment AI. You evaluate if text is fluent and coherent. Provide JSON output only with keys 'score' (float 0.0-1.0) and 'reason' (string)."},
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
            if isinstance(llm_result, dict) and 'score' in llm_result and 'reason' in llm_result:
                 # Ensure score is a float and is within the 0-1 range
                try:
                    score = float(llm_result['score'])
                    llm_result['score'] = max(0.0, min(1.0, score)) # Clamp score between 0 and 1
                    llm_result['reason'] = f"LLM Fluency: {llm_result['reason']}" # Add prefix for clarity
                    return llm_result
                except (ValueError, TypeError):
                     return {'score': 0.0, 'reason': f'LLM Fluency: Returned invalid score type in JSON: {response_content}'}
            else:
                 return {'score': 0.0, 'reason': f'LLM Fluency: Returned invalid JSON format: {response_content}'}
        except json.JSONDecodeError:
            return {'score': 0.0, 'reason': f'LLM Fluency: Returned non-JSON response: {response_content}'}
        except TypeError: # Handle cases where json.loads gets unexpected types
             return {'score': 0.0, 'reason': f'LLM Fluency: Error processing LLM response: {response_content}'}

    except Exception as e:
        # Catch potential API errors or other issues
        error_message = str(e)
        # Be careful not to expose sensitive info in error messages if this runs in production
        print(f"  LLM Fluency API Error: {error_message[:100]}...") # Log a truncated error
        return {'score': 0.0, 'reason': f"LLM Fluency Error: API call failed"} # Generic error for result


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
    Attempt to provide a very brief summary (1-2 sentences) of the following text snippet.

    If the text is substantially garbled, nonsensical, corrupted, or clearly not standard prose making summarization impossible or meaningless, please respond ONLY with the JSON object:
    {{"success": false, "summary": "Text is garbled/unsuitable for summarization."}}

    If the text is reasonably coherent and summarizable, respond ONLY with the JSON object:
    {{"success": true, "summary": "YOUR_1_TO_2_SENTENCE_SUMMARY_HERE"}}

    Text Snippet:
    ---
    {text_snippet[:2000]} # Limit snippet length for prompt
    ---
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes text if possible. If not, you state it's unsuitable. Respond ONLY with the specified JSON format: {\"success\": boolean, \"summary\": string}."},
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
            if isinstance(llm_result, dict) and 'success' in llm_result and 'summary' in llm_result and isinstance(llm_result['success'], bool):
                reason_prefix = "LLM Summarization:"
                reason = f"{reason_prefix} Summarized successfully" if llm_result['success'] else f"{reason_prefix} Failed/unsuitable"
                llm_result['reason'] = reason # Add reason field for consistency
                return llm_result
            else:
                 # Fallback if JSON is invalid or missing keys
                 return {'success': False, 'summary': '', 'reason': f'LLM Summarization: Returned invalid JSON format: {response_content}'}
        except json.JSONDecodeError:
             return {'success': False, 'summary': '', 'reason': f'LLM Summarization: Returned non-JSON response: {response_content}'}
        except TypeError:
             return {'success': False, 'summary': '', 'reason': f'LLM Summarization: Error processing LLM response: {response_content}'}


    except Exception as e:
        error_message = str(e)
        print(f"  LLM Summarization API Error: {error_message[:100]}...") # Log truncated error
        return {'success': False, 'summary': '', 'reason': f"LLM Summarization Error: API call failed"}


# --- Combined Analysis Function ---
# This function will now call analyze_text_quality which is imported
def comprehensive_text_analysis(text, client, num_llm_samples=2, **statistical_params):
    """
    Performs both statistical and LLM-based text quality analysis.

    Args:
        text (str): The full text extracted from the PDF.
        client: The initialized Groq client (can be None).
        num_llm_samples (int): Number of text snippets to sample for LLM checks.
        **statistical_params: Keyword arguments for the statistical analysis
                               (e.g., char_threshold, space_ratio_threshold, etc.).

    Returns:
        dict: Combined analysis results.
    """
    # Call the imported analyze_text_quality function
    statistical_results = analyze_text_quality(text, **statistical_params)

    # Initialize combined results with statistical results
    combined_results = statistical_results.copy()
    combined_results['llm_status'] = "Skipped (LLM client not initialized or no text)"
    # Initialize overall garbled and reason based on statistical results initially
    # Use .get() for safety in case statistical_analyzer changes its output keys
    combined_results['is_garbled'] = statistical_results.get('is_garbled_statistical', True) # Default to garbled if key missing
    combined_results['reason'] = statistical_results.get('statistical_reason', 'Statistical analysis error')
    combined_results['fluency_checks'] = []
    combined_results['summarization_checks'] = []
    combined_results['llm_samples_run'] = 0 # Track how many samples were actually checked by LLM


    if client and text and text.strip():
        # Use the imported sampling function; it handles cases with little text
        # Use a reasonable sample length for LLM context
        text_snippets = sample_text_from_string(text, num_samples=num_llm_samples, sample_length=1500)

        if not text_snippets:
             combined_results['llm_status'] = "Skipped (No usable text snippets generated)"
        else:
            combined_results['llm_status'] = "Performed"
            combined_results['llm_samples_run'] = len(text_snippets)

            # Run Fluency Check on samples
            fluency_checks_results = []
            print(f"  Running {len(text_snippets)} LLM Fluency Check(s)...")
            for snippet in text_snippets:
                 fluency_checks_results.append(check_fluency_with_llm(snippet, client))
            combined_results['fluency_checks'] = fluency_checks_results # Store detailed results

            # Run Summarization Check on samples
            summarization_checks_results = []
            print(f"  Running {len(text_snippets)} LLM Summarization Check(s)...")
            for snippet in text_snippets:
                 summarization_checks_results.append(check_summarization_with_llm(snippet, client))
            combined_results['summarization_checks'] = summarization_checks_results # Store detailed results

            # --- Aggregate LLM results to influence the overall 'is_garbled' flag ---
            # Filter out errors/skipped checks before calculating averages/ratios
            valid_fluency_scores = [
                c['score'] for c in fluency_checks_results
                if isinstance(c, dict) and 'score' in c and isinstance(c['score'], (int, float))
            ]
            avg_fluency_score = sum(valid_fluency_scores) / len(valid_fluency_scores) if valid_fluency_scores else 0.0

            valid_summarization_results = [
                c for c in summarization_checks_results
                if isinstance(c, dict) and 'success' in c and isinstance(c['success'], bool)
            ]
            summarization_failures = sum(not c['success'] for c in valid_summarization_results)
            total_valid_llm_summaries = len(valid_summarization_results)

            # Define thresholds for LLM flagging (can adjust these based on testing)
            LLM_FLUENCY_THRESHOLD = 0.5 # If average fluency score is below this, flag
            # If more than a certain ratio of valid samples fail summarization, flag
            LLM_SUMMARIZATION_FAILURE_THRESHOLD_RATIO = 0.5 # e.g., > 50% samples fail

            llm_reasons = []
            llm_flagged_garbled = False

            # Apply LLM thresholds only if we have valid results
            if valid_fluency_scores: # Check if we got any valid fluency scores
                if avg_fluency_score < LLM_FLUENCY_THRESHOLD:
                    llm_flagged_garbled = True
                    llm_reasons.append(f"LLM: Avg Fluency ({avg_fluency_score:.2f}) < {LLM_FLUENCY_THRESHOLD}")

            if total_valid_llm_summaries > 0: # Check if we got any valid summarization results
                failure_ratio = summarization_failures / total_valid_llm_summaries
                if failure_ratio > LLM_SUMMARIZATION_FAILURE_THRESHOLD_RATIO:
                    llm_flagged_garbled = True
                    llm_reasons.append(f"LLM: Summarization Failed ({summarization_failures}/{total_valid_llm_summaries} > {LLM_SUMMARIZATION_FAILURE_THRESHOLD_RATIO*100:.0f}%)")

            # --- Update overall 'is_garbled' and 'reason' based on combined checks ---
            # Logic: If EITHER statistical OR LLM checks flag it as garbled, mark as garbled.
            if llm_flagged_garbled:
                combined_results['is_garbled'] = True
                # Combine reasons, starting with statistical, then adding LLM reasons if any
                combined_results['reason'] = statistical_results.get('statistical_reason', 'Stat Err') + "; " + ", ".join(llm_reasons)
            elif not combined_results['is_garbled']: # If stats were OK and LLM didn't flag
                 combined_results['reason'] = statistical_results.get('statistical_reason', 'Stat OK') + "; LLM checks passed thresholds."
            # Else (stats flagged, LLM didn't), keep the original statistical reason and is_garbled=True


    # Add aggregated LLM scores for reporting, handle cases with no valid results
    combined_results['avg_llm_fluency'] = avg_fluency_score if 'avg_fluency_score' in locals() else None
    combined_results['llm_summary_failures'] = summarization_failures if 'summarization_failures' in locals() else None
    combined_results['llm_summary_total_valid'] = total_valid_llm_summaries if 'total_valid_llm_summaries' in locals() else 0


    return combined_results


# --- Main function to check PDFs in a directory ---
def check_pdf_directory(directory_path, client, num_llm_samples=2, **analysis_params):
    """
    Checks all PDF files in a given directory using comprehensive analysis.

    Args:
        directory_path (str): The path to the directory containing PDF files.
        client: The initialized Groq client (can be None).
        num_llm_samples (int): Number of text snippets to sample for LLM checks per PDF.
        **analysis_params: Keyword arguments to pass to the imported analyze_text_quality.

    Returns:
        list: A list of dictionaries, each containing the file path and its analysis results.
    """
    results = []
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files in '{directory_path}'.")
    # Use a consistent order for processing files
    for filename in sorted(pdf_files):
        pdf_path = os.path.join(directory_path, filename)
        print(f"--- Checking: {filename} ---") # Use filename for cleaner output

        # Extract text using the imported utility function
        extracted_text = extract_text_from_pdf(pdf_path) # Imported function

        analysis_result = {} # Initialize result dict

        if extracted_text is None:
             print(f"  ERROR: Failed to extract text from {filename}.")
             # Create a minimal result indicating extraction failure
             analysis_result = {
                 'is_garbled': True, # Treat extraction failure as garbled
                 'reason': 'Text extraction failed',
                 'llm_status': 'Skipped (Extraction failed)',
                 # Include keys expected by reporting, with default values
                 'is_garbled_statistical': True,
                 'statistical_reason': 'Text extraction failed',
                 'alphanumeric_ratio': 0.0,
                 'printable_ratio': 0.0,
                 'short_line_ratio': 0.0,
                 'avg_llm_fluency': None,
                 'llm_summary_failures': None,
                 'llm_summary_total_valid': 0
             }
        elif not extracted_text.strip():
             print(f"  WARNING: Extracted text is empty for {filename}.")
             # Create a minimal result indicating empty text
             analysis_result = {
                 'is_garbled': True, # Treat empty text as garbled
                 'reason': 'Extracted text is empty',
                 'llm_status': 'Skipped (Empty text)',
                 'is_garbled_statistical': True,
                 'statistical_reason': 'Extracted text is empty',
                 'alphanumeric_ratio': 0.0,
                 'printable_ratio': 0.0,
                 'short_line_ratio': 0.0,
                 'avg_llm_fluency': None,
                 'llm_summary_failures': None,
                 'llm_summary_total_valid': 0
             }
        else:
            # Perform comprehensive analysis using the imported statistical analyzer
            analysis_result = comprehensive_text_analysis(
                extracted_text,
                client, # Pass the client here
                num_llm_samples=num_llm_samples, # Pass the number of samples
                **analysis_params # Pass statistical thresholds here
            )
            print(f"  Overall Result: {'GARBLED' if analysis_result.get('is_garbled', True) else 'OK'}")
            print(f"  Reason: {analysis_result.get('reason', 'N/A')}")


        results.append({'file': filename, 'result': analysis_result})
        print("-" * (len(filename) + 16) + "\n") # Separator line


    return results


# --- Script Entry Point ---
if __name__ == "__main__":
    # --- Configuration ---
    # Directory containing PDFs relative to the script location
    PDF_DIRECTORY = "./pdfs_to_check"  # <--- IMPORTANT: Ensure this directory exists

    # Output CSV filename (timestamped)
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_CSV = f"pdf_analysis_results_{TIMESTAMP}.csv"

    # --- Analysis Thresholds (Adjust as needed) ---
    # Statistical Thresholds (passed to the imported analyze_text_quality)
    STATISTICAL_PARAMS = {
        "min_line_chars": 10,           # Used by statistical_analyzer
        "short_line_ratio_threshold": 0.25, # Used by statistical_analyzer
        "alphanumeric_ratio_threshold": 0.5, # Used by statistical_analyzer
        "printable_ratio_threshold": 0.7     # Used by statistical_analyzer
        # Add any other parameters your statistical_analyzer function accepts
    }

    # LLM Check Parameters
    NUM_LLM_SAMPLES_PER_PDF = 2 # How many snippets to check with LLM per PDF
    # Note: LLM sample length is defined within sample_text_from_string (imported)
    # Note: LLM decision thresholds are defined within comprehensive_text_analysis

    # ---------------------

    # Ensure the PDF directory exists
    if not os.path.isdir(PDF_DIRECTORY):
        print(f"Error: Directory not found: '{PDF_DIRECTORY}'")
        print("Please create the directory and place your PDF files inside.")
        exit(1) # Exit if directory doesn't exist

    # Check Groq client status (already initialized above)
    if groq_client is None:
        print("\nWarning: Groq client not available. LLM checks will be skipped.")
    else:
        print("\nGroq client initialized. LLM checks will be performed.")

    print(f"\nStarting text quality check for PDFs in: {PDF_DIRECTORY}")
    print(f"Statistical Params: {STATISTICAL_PARAMS}")
    print(f"LLM Samples per PDF: {NUM_LLM_SAMPLES_PER_PDF}")
    print("-" * 40)


    # Run the check_pdf_directory function
    analysis_results = check_pdf_directory(
        PDF_DIRECTORY,
        groq_client, # Pass the potentially None Groq client
        num_llm_samples=NUM_LLM_SAMPLES_PER_PDF, # Pass LLM sample count
        **STATISTICAL_PARAMS # Pass the statistical thresholds dictionary
    )

    # --- Prepare Data for Table and CSV ---
    if not analysis_results:
        print("No PDF files found or processed in the directory.")
    else:
        table_data = []
        # Define headers based on the keys available in the 'result' dictionary
        # It's safer to define expected headers explicitly
        headers = [
            "File Name",
            "Overall Garbled",
            "Combined Reason",
            "Stat Garbled",
            "Stat Reason",
            "Alpha Ratio",
            "Print Ratio",
            "Short Line Ratio",
            "LLM Status",
            "LLM Samples Run",
            "Avg Fluency",
            "Summary Failures",
            "Summary Checks"
        ]

        # Sort results by overall status (Garbled first), then filename
        sorted_results = sorted(analysis_results, key=lambda x: (not x['result'].get('is_garbled', True), x['file']))

        for item in sorted_results:
            file = item['file']
            result = item['result'] # This is the dictionary returned by comprehensive_text_analysis

            # Extract data using .get() for safety, providing defaults
            overall_garbled = result.get('is_garbled', True)
            combined_reason = result.get('reason', 'N/A')
            stat_garbled = result.get('is_garbled_statistical', True)
            stat_reason = result.get('statistical_reason', 'N/A')
            alpha_ratio = result.get('alphanumeric_ratio', 0.0)
            print_ratio = result.get('printable_ratio', 0.0)
            short_line_ratio = result.get('short_line_ratio', 0.0)
            llm_status = result.get('llm_status', 'N/A')
            llm_samples_run = result.get('llm_samples_run', 0)
            avg_fluency = result.get('avg_llm_fluency') # Can be None
            summary_failures = result.get('llm_summary_failures') # Can be None
            summary_total = result.get('llm_summary_total_valid', 0)

            # Format LLM results for display
            avg_fluency_str = f"{avg_fluency:.2f}" if avg_fluency is not None else "N/A"
            summary_failures_str = f"{summary_failures}/{summary_total}" if summary_failures is not None else "N/A"


            table_data.append([
                file,
                "YES" if overall_garbled else "NO",
                combined_reason,
                "YES" if stat_garbled else "NO",
                stat_reason,
                f"{alpha_ratio:.2f}",
                f"{print_ratio:.2f}",
                f"{short_line_ratio:.2f}",
                llm_status,
                llm_samples_run,
                avg_fluency_str,
                summary_failures_str,
                summary_total # Show total valid checks run for context
            ])

        # --- Write Data to CSV ---
        try:
            with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(headers) # Write the header row
                writer.writerows(table_data) # Write the data rows
            print(f"\nAnalysis results successfully exported to {OUTPUT_CSV}")
        except IOError as e:
            print(f"\nError writing CSV file {OUTPUT_CSV}: {e}")
        except Exception as e:
             print(f"\nAn unexpected error occurred while writing CSV: {e}")


        # --- Print the Table to Console ---
        try:
            # Ensure tabulate is imported (it should be at the top)
            print("\n" + "="*60)
            print("--- TEXT QUALITY ANALYSIS SUMMARY (Console View) ---")
            print("="*60 + "\n")
            # Adjust maxcolwidths as needed for readability
            print(tabulate(table_data, headers=headers, tablefmt="grid", maxcolwidths=[None, None, 40, None, 30, None, None, None, None, None, None, None, None]))
            print("\n" + "="*60)
        except NameError:
             print("\nTabulate library not found or not imported correctly. Skipping console table output.")
        except Exception as e:
             print(f"\nAn unexpected error occurred while printing the table: {e}")


        # --- Print Summary Lists ---
        garbled_files = [item['file'] for item in sorted_results if item['result'].get('is_garbled', True)]
        ok_files = [item['file'] for item in sorted_results if not item['result'].get('is_garbled', True)]

        print("\n--- FINAL CLASSIFICATION ---")
        print(f"\nPotential Garbled Files ({len(garbled_files)}):")
        if garbled_files:
            for f in garbled_files: print(f"  - {f}")
        else: print("  None")

        print(f"\nFiles Looking OK ({len(ok_files)}):")
        if ok_files:
             for f in ok_files: print(f"  - {f}")
        else: print("  None")
        print("\n" + "="*30 + "\n")