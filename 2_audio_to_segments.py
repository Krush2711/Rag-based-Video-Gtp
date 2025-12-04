import whisper
import json
import os

model = whisper.load_model("large-v2")
print("model loaded")

# 
# print(res)

audios = os.listdir("audios")

for audio in audios:
    num = audio.split(" ")[0]
    res = model.transcribe(audio= "outputfile.mp3", language='en')
    chunks = []
    for segment in res['segments']:
        chunks.append({"number":1, "start:":segment["start"],"end":segment["end"], "text":segment["text"]})

    chunks_with_meta_data = {"chunks":chunks, "text" : res["text"]}
    print(chunks)

    with open("output.json", "w") as f:
        json.dump(chunks_with_meta_data, f)
