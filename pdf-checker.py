import fitz # PyMuPDF
import string
import os
import textwrap # Useful for sampling text
from groq import Groq # Import the Groq client

# --- Groq Configuration ---
# It's recommended to set GROQ_API_KEY environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192" # Or another suitable Groq model
LLM_TEMP = 0.1 # Lower temperature for more consistent outputs

def get_groq_client():
    """Initializes and returns the Groq client."""
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY environment variable not set.")
        print("Please set it before running the script.")
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        return client
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        return None

# Initialize client globally or pass it around
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

def sample_text_from_string(text, num_samples=3, sample_length=500):
    """
    Samples text snippets from a larger string.

    Args:
        text (str): The source text.
        num_samples (int): The number of snippets to sample.
        sample_length (int): The desired length of each snippet.

    Returns:
        list: A list of text snippets.
    """
    if not text or len(text) < sample_length:
        return [text[:sample_length]] # Return what's available
    
    samples = []
    text_length = len(text)
    # Calculate step size to get roughly even distribution
    step = max(1, text_length // num_samples)

    for i in range(num_samples):
        start_index = min(i * step, text_length - sample_length)
        end_index = start_index + sample_length
        samples.append(text[start_index:end_index])

    return samples


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
    if not client or not text_snippet or not text_snippet.strip():
        return {'score': 0.0, 'reason': 'No text snippet provided or client not initialized'}

    prompt = f"""
    Analyze the following text snippet for its fluency, coherence, and whether it appears to be standard human-readable text (like from a document or book) versus being garbled, corrupted, or non-sensical data.

    Provide a score between 0.0 (completely garbled/unreadable) and 1.0 (perfectly fluent and coherent). Also, provide a brief reason for your score.

    Format your output as a JSON object with keys "score" (float) and "reason" (string).

    Text Snippet:
    ---
    {text_snippet[:2000]} # Limit snippet length for prompt
    ---
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a text quality assessment AI. You evaluate if text is fluent and coherent."},
                {"role": "user", "content": prompt},
            ],
            model=GROQ_MODEL,
            temperature=LLM_TEMP,
            response_format={"type": "json_object"} # Request JSON output
        )
        response_content = chat_completion.choices[0].message.content
        # Attempt to parse the JSON response
        import json
        try:
            llm_result = json.loads(response_content)
            if 'score' in llm_result and 'reason' in llm_result:
                 # Ensure score is a float
                llm_result['score'] = float(llm_result['score'])
                return llm_result
            else:
                 return {'score': 0.0, 'reason': f'LLM returned invalid JSON format: {response_content}'}
        except json.JSONDecodeError:
            return {'score': 0.0, 'reason': f'LLM returned non-JSON response: {response_content}'}

    except Exception as e:
        return {'score': 0.0, 'reason': f"Error calling LLM (Fluency): {e}"}


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
    if not client or not text_snippet or not text_snippet.strip():
         return {'success': False, 'summary': '', 'reason': 'No text snippet provided or client not initialized'}

    prompt = f"""
    Attempt to provide a very brief summary (1-2 sentences) of the following text snippet. If the text is garbled, nonsensical, or cannot be summarized, please state clearly that it cannot be summarized and explain why.

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

        # Simple check for indicators of failure
        failure_indicators = [
            "cannot be summarized",
            "unable to summarize",
            "text is garbled",
            "text is nonsensical",
            "difficult to understand",
            "appears to be random characters"
        ]
        is_successful = not any(indicator in response_content.lower() for indicator in failure_indicators)

        return {'success': is_successful, 'summary': response_content, 'reason': 'Summarized successfully' if is_successful else 'LLM indicated inability to summarize'}

    except Exception as e:
         return {'success': False, 'summary': '', 'reason': f"Error calling LLM (Summarization): {e}"}


# --- Update analyze_text_quality or create a new function to include LLM checks ---

# We will create a new function to combine statistical and LLM analysis
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
    statistical_results = analyze_text_quality(text, **statistical_params)

    llm_results = {}
    if client and text and text.strip():
        text_snippets = sample_text_from_string(text, num_samples=num_llm_samples, sample_length=1000) # Using larger snippets for LLM
        llm_results['llm_samples'] = text_snippets

        # Run Fluency Check on samples
        fluency_checks = []
        for i, snippet in enumerate(text_snippets):
             print(f"  Running LLM Fluency Check on sample {i+1}/{num_llm_samples}...")
             fluency_checks.append(check_fluency_with_llm(snippet, client))
        llm_results['fluency_checks'] = fluency_checks

        # Run Summarization Check on samples
        summarization_checks = []
        for i, snippet in enumerate(text_snippets):
             print(f"  Running LLM Summarization Check on sample {i+1}/{num_llm_samples}...")
             summarization_checks.append(check_summarization_with_llm(snippet, client))
        llm_results['summarization_checks'] = summarization_checks

        # Aggregate LLM results to influence the overall 'is_garbled' flag
        # Simple aggregation: if any sample strongly indicates garbling by LLM
        avg_fluency_score = sum(c['score'] for c in fluency_checks) / len(fluency_checks) if fluency_checks else 1.0
        summarization_failures = sum(not c['success'] for c in summarization_checks)

        # Define thresholds for LLM flagging
        LLM_FLUENCY_THRESHOLD = 0.5 # If average fluency score is below this, flag
        LLM_SUMMARIZATION_FAILURE_THRESHOLD = num_llm_samples * 0.5 # If more than 50% of summaries fail, flag

        if avg_fluency_score < LLM_FLUENCY_THRESHOLD:
            statistical_results['is_garbled'] = True
            statistical_results['reason'] += f", LLM (Avg Fluency: {avg_fluency_score:.2f})"
        if summarization_failures > LLM_SUMMARIZATION_FAILURE_THRESHOLD:
             statistical_results['is_garbled'] = True
             statistical_results['reason'] += f", LLM ({summarization_failures}/{num_llm_samples} Summaries Failed)"

    else:
        llm_results['llm_status'] = "Skipped (Groq client not initialized or no text)"
        # If LLM checks skipped due to no client/key, don't override statistical results
        # If skipped due to no text, statistical results already marked as garbled

    # Combine results
    combined_results = {**statistical_results, **llm_results}

    return combined_results


# --- Update the main check_pdf_directory function ---
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
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            print(f"Checking: {pdf_path}")
            extracted_text = extract_text_from_pdf(pdf_path)

            # Pass the Groq client and num_llm_samples to the comprehensive analysis
            analysis_result = comprehensive_text_analysis(
                extracted_text,
                client, # Pass the client here
                num_llm_samples=num_llm_samples, # Pass the number of samples
                **analysis_params
            )
            results.append({'file': filename, 'result': analysis_result})


    return results

if __name__ == "__main__":
    # Ensure the Groq client was initialized successfully
    if groq_client is None and os.environ.get("GROQ_API_KEY"):
        print("Groq client failed to initialize. LLM checks will be skipped.")
    elif groq_client is None:
        print("GROQ_API_KEY not set. LLM checks will be skipped.")
    else:
        print("Groq client initialized successfully. LLM checks will be performed.")


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
    llm_sample_length = 1500 # Approx chars per sample (adjust based on token limits/cost)

    # LLM Thresholds (within comprehensive_text_analysis function)
    # LLM_FLUENCY_THRESHOLD = 0.5
    # LLM_SUMMARIZATION_FAILURE_THRESHOLD = num_llm_samples * 0.5 # Defined inside analysis function


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

        print("\n--- Summary ---")
        garbled_files = []
        ok_files = []

        for item in analysis_results:
            file = item['file']
            result = item['result']
            print(f"File: {file}")
            print(f"  Total Chars: {result.get('total_chars', 'N/A')}")
            print(f"  Total Lines: {result.get('total_lines', 'N/A')}")
            print(f"  Alphanumeric Ratio: {result.get('alphanumeric_ratio', 0.0):.2f}")
            print(f"  Printable Ratio:    {result.get('printable_ratio', 0.0):.2f}")
            print(f"  Short Line Ratio:   {result.get('short_line_ratio', 0.0):.2f}")

            # Print LLM results if available
            if 'llm_status' in result:
                 print(f"  LLM Checks: {result['llm_status']}")
            elif 'fluency_checks' in result:
                print("  LLM Fluency Checks:")
                for i, fc in enumerate(result['fluency_checks']):
                    print(f"    Sample {i+1}: Score={fc.get('score', 0.0):.2f}, Reason: {fc.get('reason', 'N/A')}")
                print("  LLM Summarization Checks:")
                for i, sc in enumerate(result['summarization_checks']):
                    print(f"    Sample {i+1}: Success={sc.get('success', False)}, Reason: {sc.get('reason', 'N/A')}")
                    # Optional: print summary itself
                    # print(f"      Summary: {sc.get('summary', 'N/A')[:100]}...") # Print first 100 chars of summary


            print(f"  Overall Is Garbled: {result.get('is_garbled', True)}")
            print(f"  Reason: {result.get('reason', 'Unknown')}\n")

            if result.get('is_garbled', True):
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