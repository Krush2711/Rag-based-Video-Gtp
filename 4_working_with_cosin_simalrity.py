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

# df = pd.read_csv('cunks_Sample.csv')

df = joblib.load('embeddings.joblib')


# df['embedding'] = df['embedding'].apply(
#     lambda x: np.array(ast.literal_eval(x), dtype=float)
# )


# print(df)

user_query = input("Ask a quesstion : ")
question_embeddings = get_embeddings([user_query])[0]

# # print(np.vstack(df['embedding']).values)
# # print(np.vstack(df['embedding']).shape)

similarity = cosine_similarity(np.vstack(df['embedding']), [question_embeddings]).flatten()
print(similarity)

print("\n?????????????????????????????????????????????????????????????\n")

top_res = 10
max_indc = similarity.argsort()[::-1][0:top_res] # give accending top reslults , ebers it by using [::-1]
print(max_indc)
print(similarity[max_indc])
most_accurate_df = df.loc[max_indc]
print("\n?????????????????????????????????????????????????????????????\n")
print(most_accurate_df[['number', 'text', 'start:']])