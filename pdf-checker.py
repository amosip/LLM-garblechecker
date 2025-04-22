#new refactored pdf_check.py script, broken out into functions and classes for better readability and maintainability

# Imports for utilities
from pdf_utils import extract_text_from_pdf, sample_text_from_string
# Import for statistical analysis
from statistical_analyzer import analyze_text_quality
# Add imports for LLM service functions
from llm_service import get_groq_client, check_fluency_with_llm, check_summarization_with_llm

# Standard library imports
from dotenv import load_dotenv
import csv
from datetime import datetime
import os # Needed for file path operations and env vars
# json is no longer strictly needed here if only LLM service functions use it,
# but keeping it doesn't hurt. If it's used elsewhere, keep the import.
# import json

# Third-party imports
# Groq is now imported within llm_service.py
# from groq import Groq
from tabulate import tabulate # Import tabulate for table formatting


# Load environment variables (should be done early)
load_dotenv()

# --- Groq Configuration (Now in llm_service.py) ---
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# GROQ_MODEL = "deepseek-r1-distill-llama-70b"
# LLM_TEMP = 0.1

# --- Groq Client Initialization (Now imported/handled by get_groq_client from llm_service) ---
# def get_groq_client(): ... (removed)
# Initialize client globally (will be None if key is missing)
# groq_client = get_groq_client() # This global initialization will be moved below load_dotenv

# --- PDF Utility Functions (Now Imported) ---
# ... (commented out) ...

# --- Statistical Analysis Function (Now Imported) ---
# ... (commented out) ...

# --- LLM Check Functions (Now in llm_service.py) ---
# def check_fluency_with_llm(...) ... (removed)
# def check_summarization_with_llm(...) ... (removed)


# --- Combined Analysis Function ---
# This function now calls imported analyze_text_quality and LLM check functions
def comprehensive_text_analysis(text, client, num_llm_samples=2, **statistical_params):
    """
    Performs both statistical and LLM-based text quality analysis.

    Args:
        text (str): The full text extracted from the PDF.
        client: The initialized Groq client instance (can be None).
        num_llm_samples (int): Number of text snippets to sample for LLM checks.
        **statistical_params: Keyword arguments for the statistical analysis.

    Returns:
        dict: Combined analysis results.
    """
    # Call the imported analyze_text_quality function
    statistical_results = analyze_text_quality(text, **statistical_params)

    # Initialize combined results with statistical results
    combined_results = statistical_results.copy()
    combined_results['llm_status'] = "Skipped (LLM client not initialized or no text)"
    combined_results['is_garbled'] = statistical_results.get('is_garbled_statistical', True)
    combined_results['reason'] = statistical_results.get('statistical_reason', 'Statistical analysis error')

    # Initialize fields for LLM results
    combined_results['fluency_checks'] = []
    combined_results['summarization_checks'] = []
    combined_results['llm_samples_run'] = 0
    combined_results['avg_llm_fluency'] = None # Default to None
    combined_results['llm_summary_failures'] = None # Default to None
    combined_results['llm_summary_total_valid'] = 0 # Default to 0

    # --- Perform LLM Checks if client is available and text exists ---
    if client and text and text.strip():
        # Use the imported sampling function
        text_snippets = sample_text_from_string(text, num_samples=num_llm_samples, sample_length=1500)

        if not text_snippets:
             combined_results['llm_status'] = "Skipped (No usable text snippets generated)"
        else:
            combined_results['llm_status'] = "Performed"
            combined_results['llm_samples_run'] = len(text_snippets)
            # Note: We are NOT storing 'llm_samples' in combined_results anymore to keep result dict cleaner

            # Run Fluency Check on samples using the imported function
            fluency_checks_results = []
            print(f"  Running {len(text_snippets)} LLM Fluency Check(s)...")
            for snippet in text_snippets:
                 fluency_checks_results.append(check_fluency_with_llm(snippet, client)) # Imported function
            combined_results['fluency_checks'] = fluency_checks_results # Store detailed results

            # Run Summarization Check on samples using the imported function
            summarization_checks_results = []
            print(f"  Running {len(text_snippets)} LLM Summarization Check(s)...")
            for snippet in text_snippets:
                 summarization_checks_results.append(check_summarization_with_llm(snippet, client)) # Imported function
            combined_results['summarization_checks'] = summarization_checks_results # Store detailed results

            # --- Aggregate LLM results for overall flag and reason ---

            valid_fluency_scores = [
                c['score'] for c in fluency_checks_results
                if isinstance(c, dict) and 'score' in c and isinstance(c['score'], (int, float))
            ]
            avg_fluency_score = sum(valid_fluency_scores) / len(valid_fluency_scores) if valid_fluency_scores else 0.0
            combined_results['avg_llm_fluency'] = avg_fluency_score # Update aggregated field


            valid_summarization_results = [
                c for c in summarization_checks_results
                if isinstance(c, dict) and 'success' in c and isinstance(c['success'], bool)
            ]
            summarization_failures = sum(not c['success'] for c in valid_summarization_results)
            total_valid_llm_summaries = len(valid_summarization_results)
            combined_results['llm_summary_failures'] = summarization_failures # Update aggregated field
            combined_results['llm_summary_total_valid'] = total_valid_llm_summaries # Update aggregated field


            # Define thresholds for LLM flagging (can adjust these based on testing)
            LLM_FLUENCY_THRESHOLD = 0.5 # If average fluency score is below this, flag
            LLM_SUMMARIZATION_FAILURE_THRESHOLD_RATIO = 0.5 # e.g., > 50% samples fail

            llm_reasons = []
            llm_flagged_garbled = False

            # Apply LLM thresholds only if we have valid results
            if valid_fluency_scores:
                 if avg_fluency_score < LLM_FLUENCY_THRESHOLD:
                    llm_flagged_garbled = True
                    llm_reasons.append(f"LLM: Avg Fluency ({avg_fluency_score:.2f} < {LLM_FLUENCY_THRESHOLD})")

            if total_valid_llm_summaries > 0:
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


    return combined_results


# --- Main function to check PDFs in a directory ---
def check_pdf_directory(directory_path, client, num_llm_samples=2, **analysis_params):
    """
    Checks all PDF files in a given directory using comprehensive analysis.

    Args:
        directory_path (str): The path to the directory containing PDF files.
        client: The initialized Groq client instance (can be None).
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

        # Handle cases where text extraction fails or returns empty text early
        if extracted_text is None:
             print(f"  ERROR: Failed to extract text from {filename}.")
             analysis_result = {
                'is_garbled': True, # Treat extraction failure as garbled
                'reason': 'Text extraction failed',
                'llm_status': 'Skipped (Extraction failed)',
                'is_garbled_statistical': True,
                'statistical_reason': 'Text extraction failed',
                'alphanumeric_ratio': 0.0, 'printable_ratio': 0.0, 'short_line_ratio': 0.0,
                'llm_samples_run': 0, 'avg_llm_fluency': None,
                'llm_summary_failures': None, 'llm_summary_total_valid': 0,
                'fluency_checks': [], 'summarization_checks': [] # Include empty lists for structure
             }
        elif not extracted_text.strip():
             print(f"  WARNING: Extracted text is empty for {filename}.")
             analysis_result = {
                'is_garbled': True, # Treat empty text as garbled
                'reason': 'Extracted text is empty',
                'llm_status': 'Skipped (Empty text)',
                'is_garbled_statistical': True,
                'statistical_reason': 'Extracted text is empty',
                'alphanumeric_ratio': 0.0, 'printable_ratio': 0.0, 'short_line_ratio': 0.0,
                'llm_samples_run': 0, 'avg_llm_fluency': None,
                'llm_summary_failures': None, 'llm_summary_total_valid': 0,
                 'fluency_checks': [], 'summarization_checks': [] # Include empty lists for structure
             }
        else:
            # Perform comprehensive analysis using the imported functions
            analysis_result = comprehensive_text_analysis(
                extracted_text,
                client, # Pass the potentially None client
                num_llm_samples=num_llm_samples,
                **analysis_params # Pass statistical thresholds
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
        "min_line_chars": 10,
        "short_line_ratio_threshold": 0.25,
        "alphanumeric_ratio_threshold": 0.7,
        "printable_ratio_threshold": 0.7
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

    # Initialize the Groq client *after* loading environment variables
    groq_client_instance = get_groq_client() # Call the imported initialization function

    # Check Groq client status
    if groq_client_instance is None:
        print("Warning: Groq client not available. LLM checks will be skipped.")
    else:
        print("Groq client initialized. LLM checks will be performed for files with sufficient text.")

    print(f"\nStarting text quality check for PDFs in: {PDF_DIRECTORY}")
    print(f"Statistical Params: {STATISTICAL_PARAMS}")
    print(f"LLM Samples per PDF: {NUM_LLM_SAMPLES_PER_PDF}")
    print("-" * 40)


    # Run the check_pdf_directory function
    analysis_results = check_pdf_directory(
        PDF_DIRECTORY,
        groq_client_instance, # Pass the potentially None Groq client instance
        num_llm_samples=NUM_LLM_SAMPLES_PER_PDF,
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
            "Summary Failures", # This column now represents X/Y Failed
            # No longer including Sample 1 Summary/Reason directly in this table for simplicity/width
            # You can still access detailed checks in the raw results if needed
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

            # Format LLM results for display from aggregated fields
            avg_fluency = result.get('avg_llm_fluency')
            avg_fluency_str = f"{avg_fluency:.2f}" if avg_fluency is not None else "N/A"

            summary_failures = result.get('llm_summary_failures')
            summary_total = result.get('llm_summary_total_valid', 0) # Use the valid total count
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
                summary_failures_str, # Use the X/Y Failed format
            ])

        # --- Write Data to CSV ---
        try:
            with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                writer.writerows(table_data)
            print(f"\nAnalysis results successfully exported to {OUTPUT_CSV}")
        except IOError as e:
            print(f"\nError writing CSV file {OUTPUT_CSV}: {e}")
        except Exception as e:
             print(f"\nAn unexpected error occurred while writing CSV: {e}")


        # --- Print the Table to Console ---
        try:
            print("\n" + "="*60)
            print("--- TEXT QUALITY ANALYSIS SUMMARY (Console View) ---")
            print("="*60 + "\n")
            # Adjusted maxcolwidths (might need further tuning)
            print(tabulate(table_data, headers=headers, tablefmt="grid", maxcolwidths=[None, None, 40, None, 30, None, None, None, None, None, None, None]))
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