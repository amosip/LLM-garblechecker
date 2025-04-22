# llm_service.py

import os
import json
from groq import Groq

# --- Groq Configuration ---
# Access API key from environment variable (loaded by main script via dotenv)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# Using the model you specified
GROQ_MODEL = "deepseek-r1-distill-llama-70b" # Or another suitable model like mixtral-8x7b-32768
LLM_TEMP = 0.1 # Lower temperature for more consistent outputs

# Initialize client instance within the module
# This client can be imported and used directly by other parts of the application
_groq_client_instance = None

def get_groq_client():
    """
    Initializes and returns the Groq client.
    Returns None if API key is not set or initialization fails.
    """
    global _groq_client_instance
    if _groq_client_instance is None:
        if not GROQ_API_KEY:
            print("Warning: GROQ_API_KEY environment variable not set. Cannot initialize Groq client.")
            return None
        try:
            _groq_client_instance = Groq(api_key=GROQ_API_KEY)
            # Optional: Test client connection (uncomment if needed)
            # print("Testing Groq client connection from llm_service...")
            # models = _groq_client_instance.models.list()
            # print(f"Successfully listed {len(models.data)} models from llm_service.")
            # print("Groq client initialized successfully.")
        except Exception as e:
            _groq_client_instance = None # Ensure it's None on failure
            print(f"Error initializing Groq client in llm_service: {e}")
            print("Please double-check your GROQ_API_KEY and network connection.")

    return _groq_client_instance


# --- LLM Check Method 1: Fluency and Coherence ---
def check_fluency_with_llm(text_snippet, client):
    """
    Checks the fluency and coherence of a text snippet using an LLM.

    Args:
        text_snippet (str): The text snippet to check.
        client: The initialized Groq client instance (can be None).

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
            model=GROQ_MODEL, # Use the module-level model constant
            temperature=LLM_TEMP, # Use the module-level temperature constant
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
        # print(f"  LLM Fluency API Error: {error_message[:100]}...") # Optional: Log a truncated error
        return {'score': 0.0, 'reason': f"LLM Fluency Error: API call failed ({error_message})"} # Include error message for debugging


# --- LLM Check Method 2: Attempted Task Performance (Summarization) ---
def check_summarization_with_llm(text_snippet, client):
    """
    Attempts to summarize a text snippet using an LLM as a check for usability.
    Success indicates usable text, failure indicates potential garbling.

    Args:
        text_snippet (str): The text snippet to attempt to summarize.
        client: The initialized Groq client instance (can be None).

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
            model=GROQ_MODEL, # Use the module-level model constant
            temperature=LLM_TEMP, # Use the module-level temperature constant
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
        # print(f"  LLM Summarization API Error: {error_message[:100]}...") # Optional: Log truncated error
        return {'success': False, 'summary': '', 'reason': f"LLM Summarization Error: API call failed ({error_message})"} # Include error message for debugging


# Optional: If you want to initialize the client immediately when the module is imported,
# you can uncomment the line below. However, initializing in the main script's
# if __name__ == "__main__": block after load_dotenv() is generally preferred
# for scripts that might be imported elsewhere.
# _groq_client_instance = get_groq_client()