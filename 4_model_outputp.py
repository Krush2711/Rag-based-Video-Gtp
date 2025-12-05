import pandas as pd 
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
import requests
import joblib

def get_embeddings(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model" : "bge-m3",
        "input" : text_list
        })
    emb = r.json()['embeddings']
    return emb


def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "max_tokens": 600
        }
    )
    return r.json()["response"]

df = joblib.load('New_embeddings.joblib')

user_query = input("Ask a quesstion : ")
question_embeddings = get_embeddings([user_query])[0]

similarity = cosine_similarity(np.vstack(df['embedding']), [question_embeddings]).flatten()



top_res = 5 
max_indc = similarity.argsort()[::-1][0:top_res] # give accending top reslults , ebers it by using [::-1]
# print(max_indc)
# print("\n?????????????????????????????????????????????????????????????\n")
# print(similarity[max_indc])
most_accurate_df = df.loc[max_indc]

# print("\n?????????????????????????????????????????????????????????????\n")


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
{most_accurate_df.reset_index()[['index','number','text','start:','end']].to_json()}

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


Model_responce = inference(prompt)
print(Model_responce)

with open("responce.txt", "w") as f:
    f.write(Model_responce)