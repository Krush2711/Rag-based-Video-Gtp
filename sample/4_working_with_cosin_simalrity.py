import pandas as pd 
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
import requests
import ast 
import joblib

def get_embeddings(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model" : "bge-m3",
        "input" : text_list
        })
    emb = r.json()['embeddings']
    return emb


def infernce(prompt):
     r = requests.post("http://localhost:11434/api/generate", json={
        "model" : "llama3.2:3b",
        "prompt" : prompt,
        "stream": False
        })
     res = r.json()
     return res

df = joblib.load('embeddings.joblib')

user_query = input("Ask a quesstion : ")
question_embeddings = get_embeddings([user_query])[0]

# # print(np.vstack(df['embedding']).values)
# # print(np.vstack(df['embedding']).shape)

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

Below are subtitle chunks from the course videos. Each chunk contains:
- video number
- start time (seconds)
- end time (seconds)
- transcript text

Subtitle Data:
{most_accurate_df[['number', 'text', 'start:', 'end']].to_json()}

------------------------------

User Question:
"{user_query}"

------------------------------

Your Task:
1. Determine whether the user's question is related to the course content.
2. If the question IS related:
    - Search the subtitle chunks.
    - Identify which video(s) contain the answer.
    - Provide:
        • Video number  
        • Timestamp range  
        • Short explanation of what is taught there
    - Guide the user to watch those timestamps.
3. If the question is NOT related to the course, respond:
    "I can only answer questions related to the Machine Learning Specialization by Andrew Ng."

Return only the answer, not the reasoning steps.
"""

# print(most_accurate_df[['number', 'text', 'start:', 'end']])

with open("Promt.txt", "w") as f:
    f.write(prompt)

Model_responce = infernce(prompt)['response']
print(Model_responce)

with open("responsce .txt", "w") as f:
    f.write(Model_responce)