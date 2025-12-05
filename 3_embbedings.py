import requests 
import os 
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def get_embeddings(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model" : "bge-m3",
        "input" : text_list
        })
    emb = r.json()['embeddings']
    return emb

my_chukns = []
chunk_id = 0

jsons = os.listdir("jsons")


for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"working with {json_file}")
    embeddings = get_embeddings([c['text'] for c in content['chunks']])
    for i, chunk in enumerate(content['chunks']):
        chunk["chunk_id"] = chunk_id
        chunk_id +=1 
        chunk["embedding"] = embeddings[i]
        my_chukns.append(chunk)
    
    



df = pd.DataFrame.from_records(my_chukns)
# print(df)
# df.to_csv("cunks_Sample.csv", index=False)

joblib.dump(df, "embeddings.joblib")
# user_query = input("Ask a quesstion : ")
# question_embeddings = get_embeddings(user_query)[0]
# print(question_embeddings)

