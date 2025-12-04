import os 
import subprocess

files = os.listdir("videos")
print(files)

for file in files :
    #print(file)
    file_num = file.split('Ma')[0].split('#')[1]
    # print(file_num)
    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{file_num}.mp3"])
    