import os
import tensorflow as tf
import numpy as np
# import pyttsx3
from gtts import gTTS

from imageai.Detection import ObjectDetection

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "image2.jpg"), output_image_path=os.path.join(execution_path, "image2new.jpg"))
for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"])

detections, extracted_images = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "image2.jpg"), output_image_path=os.path.join(execution_path, "image2new.jpg"), extract_detected_objects=True)

for eachObject in detections:
  Outfile = open('output.txt', 'a+')
  Outfile.write(eachObject["name"])
  Outfile.write(" ")
  Outfile.close()

print(" ")
print("Total No. of objects present")

file = open("output.txt", "r+")
# file.flush()
wordcount = {}
# for word in wordcount.items():
for word in file.read().split():
    if word not in wordcount:
        wordcount[word] = 1
    else:
        wordcount[word] += 1
for item in wordcount.items():
  
  print("{}\t{}".format(*item))

# print("{}\t{}".format(*word))
# print (word,wordcount)
file.truncate(0)
file.close()
f1 = open("voice.txt", "w")
for item in wordcount.items():
   f1.write("{}\t{}".format(*item)+"\n")
f1.close()


infile = "voice.txt"
f = open(infile, 'r')
theText = f.read()
f.close()

# Saving part starts from here
tts = gTTS(text=theText, lang='en', slow=True)
tts.save("saved_file.mp3")
os.system("saved_file.mp3")
print("File saved!")
file = open("voice.txt", "r+")
file.truncate(0)
file.close()
