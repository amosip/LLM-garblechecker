#!/usr/bin/env python3
"""
evaluate_pdf_for_rag.py

Extracts the text layer from a scientific/academic PDF and uses the Groq API
to determine if it’s suitable for inclusion in a vector database for RAG.
"""

import os
import sys

from groq import Groq
import PyPDF2

# -----------------------------------------------------------------------------
# 1) Your system prompt from before
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a document quality analysis agent tasked with evaluating whether the text layer of a given scientific or academic PDF document is suitable for inclusion in a vector database used for RAG-based search and retrieval.
Your judgment should be grounded in the document's fitness for semantic chunking, embedding, and accurate retrieval, with minimal hallucination risk.

Return your assessment in JSON:
{
  "suitability": "SUITABLE" | "UNSUITABLE",
  "issues": ["Brief summary of identified issues"],
  "confidence": "High" | "Medium" | "Low",
  "rationale": "Detailed explanation of reasoning behind decision."
}
"""

# -----------------------------------------------------------------------------
# 2) PDF text extraction
# -----------------------------------------------------------------------------
def extract_text_from_pdf(path: str) -> str:
    """
    Reads a PDF file and concatenates the text from all pages.
    """
    reader = PyPDF2.PdfReader(path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n\n".join(text)

# -----------------------------------------------------------------------------
# 3) Call Groq API
# -----------------------------------------------------------------------------
def evaluate_text(text: str, client: Groq) -> dict:
    """
    Sends the system prompt and document text to Groq for evaluation.
    Returns the parsed JSON response.
    """
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ],
    )
    # The Groq client returns a Pydantic model; convert to dict
    return response.choices[0].message.content

# -----------------------------------------------------------------------------
# 4) Main entrypoint
# -----------------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.isfile(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    # Extract
    print("Extracting text from PDF...")
    doc_text = extract_text_from_pdf(pdf_path)

    # Initialize Groq client (ensure GROQ_API_KEY is set in your environment)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: Please set the GROQ_API_KEY environment variable.")
        sys.exit(1)

    client = Groq(api_key=api_key)

    # Evaluate
    print("Sending to Groq for analysis...")
    result = evaluate_text(doc_text, client)

    # Output
    print("\n=== Groq Evaluation Result ===")
    print(result)

if __name__ == "__main__":
    main()
