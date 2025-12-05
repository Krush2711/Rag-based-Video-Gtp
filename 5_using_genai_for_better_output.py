import pandas as pd 
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
import requests
import joblib
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import sys

# --- CONFIGURATION ---
# Ensure you have your API key set in config.py or replace this line
try:
    from config import GEMINI_API_KEY
except ImportError:
    print("Error: config.py not found. Please set your GEMINI_API_KEY.")
    sys.exit(1)

# --- HELPER FUNCTIONS ---

def get_embeddings(text_list):
    """
    Fetches embeddings from a local Ollama instance or similar.
    Ensure Ollama is running: `ollama serve`
    """
    try:
        r = requests.post("http://localhost:11434/api/embed", json={
            "model" : "bge-m3",
            "input" : text_list
        })
        r.raise_for_status() # Raise error for bad responses
        emb = r.json()['embeddings']
        return emb
    except Exception as e:
        print(f"Error fetching embeddings: {e}")
        # Return a dummy zero vector to prevent crash during debugging if Ollama isn't on
        # Assuming 1024 dim for bge-m3, adjust as needed
        return [np.zeros(1024)] 

def get_available_model():
    """
    Dynamically finds a working Gemini model to avoid 404 errors.
    Prioritizes 2.5 Flash, then 1.5 Flash.
    """
    preferred_models = ["gemini-2.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-pro"]
    
    try:
        # List all available models properly
        available_models = [m.name.replace('models/', '') for m in genai.list_models()]
        
        # Check for our preferred models in the available list
        for pref in preferred_models:
            if pref in available_models:
                return pref
            # Check for versioned matches (e.g., gemini-1.5-flash-001)
            for avail in available_models:
                if pref in avail:
                    return avail
                    
        return "gemini-1.5-flash" # Fallback
    except Exception as e:
        print(f"Warning: Could not list models ({e}). Defaulting to gemini-2.5-flash.")
        return "gemini-2.5-flash"

def inference(prompt):
    # 1. Configure API
    genai.configure(api_key=GEMINI_API_KEY)
    
    # 2. Select Model (Dynamically or Hardcoded)
    model_name = get_available_model()
    print(f"Using Model: {model_name}")

    try:
        model = genai.GenerativeModel(model_name)
        
        # 3. Set Safety Settings (Reduce False Positives)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "top_p": 0.9,
                "max_output_tokens": 2000 # Increased to prevent 'FinishReason: 2'
            },
            safety_settings=safety_settings
        )
        
        # 4. Robust Text Extraction
        # Handle cases where response is blocked or truncated
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                return candidate.content.parts[0].text
            else:
                # If text is missing, check why
                return f"Error: No text generated. Finish Reason: {candidate.finish_reason}"
        elif response.prompt_feedback:
             return f"Error: Prompt Blocked. Feedback: {response.prompt_feedback}"
        else:
             return "Error: Empty response from API."

    except Exception as e:
        return f"Error Generating Content: {e}"

# --- MAIN EXECUTION ---

# 1. Load Data
try:
    df = joblib.load('embeddings.joblib')
except FileNotFoundError:
    print("Error: 'embeddings.joblib' file not found. Please generate it first.")
    sys.exit(1)

# 2. Get User Input
user_query = input("Ask a question: ")

if not user_query.strip():
    print("Empty question provided. Exiting.")
    sys.exit()

# 3. Calculate Similarity
try:
    # Ensure get_embeddings returns a valid list
    query_emb_response = get_embeddings([user_query])
    if not query_emb_response:
        raise ValueError("Empty embedding received")
        
    question_embeddings = query_emb_response[0]
    
    # Calculate Cosine Similarity
    similarity = cosine_similarity(np.vstack(df['embedding']), [question_embeddings]).flatten()

    # 4. Filter Top Results
    top_res = 5 
    max_indc = similarity.argsort()[::-1][0:top_res]
    most_accurate_df = df.loc[max_indc]

    # 5. Prepare Prompt
    # FIX: Dynamically detect column names to handle 'start' vs 'start:' mismatch
    df_cols = most_accurate_df.columns
    start_col = 'start' if 'start' in df_cols else 'start:'
    end_col = 'end' if 'end' in df_cols else 'end:'

    # Check if we found valid columns, otherwise let the user know what's available
    if start_col not in df_cols:
        print(f"\nError: Could not find 'start' or 'start:' column.")
        print(f"Available columns: {df_cols.tolist()}")
        sys.exit(1)

    # Reset index to get the chunk_id as a column
    subset = most_accurate_df.reset_index()
    
    # Handle index column name (reset_index usually creates 'index', but let's be safe)
    index_col = 'index' if 'index' in subset.columns else subset.columns[0]
    
    # Extract data using the detected column names
    # We rename them to 'start' and 'end' in the JSON so the LLM prompt remains consistent
    json_context = subset[[index_col, 'number', 'text', start_col, end_col]].rename(
        columns={start_col: 'start', end_col: 'end'}
    ).to_json()

    prompt = f"""
    You are assisting a student studying the Machine Learning Specialization by Andrew Ng.

    Below are subtitle chunks from the course. Each row contains:
    - chunk_id (this is the DataFrame index — NOT the video number)
    - number = the actual video number
    - start = start time in seconds
    - end = end time in seconds
    - text = transcript text

    IMPORTANT:
    • Only use the column 'number' as the video number.
    • Never use the DataFrame index as a video number. The index is only a chunk ID.

    Subtitle Data:
    {json_context}

    ------------------------------

    User Question:
    "{user_query}"

    ------------------------------

    Your Task:
    1. First check if the question is related to the course.
    2. If related:
       - Search through the chunks.
       - Identify which video (using ONLY the 'number' column) contains the answer.
       - Report:
         • video number  
         • timestamp range (start–end)  
         • short explanation  
    3. If unrelated:
       Reply: "I can only answer questions related to the Machine Learning Specialization by Andrew Ng."

    Return only the answer, no reasoning steps.
    """

    # 6. Run Inference
    print("\nGenerating answer...")
    Model_responce = inference(prompt)
    print("\n--- Answer ---")
    print(Model_responce)

    # 7. Save Output
    with open("response_genai.txt", "w", encoding='utf-8') as f:
        f.write(Model_responce)
        print("\nSaved response to response_genai.txt")

except KeyError as e:
    print(f"\nData Error: Column not found in DataFrame - {e}")
    print("Please check your 'embeddings.joblib' column names. (e.g., is it 'start' or 'start:'?)")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")