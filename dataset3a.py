from google.colab import files
!pip install gTTS
from gtts import gTTS
import os
!pip install datasets huggingface_hub
from datasets import load_dataset
from google.colab import files
# Load a classic dataset (e.g., SQuAD for question answering)
dataset = load_dataset("LangAGI-Lab/cactus")

print(dataset)
# Access the training split
train_data = dataset["train"]
!pip install edge-tts
!edge-tts --list-voices

import edge_tts
from IPython.display import Audio

async def main():
    """Main function"""
    communicate = edge_tts.Communicate(TEXT, VOICE, rate=RATE)
    await communicate.save(OUTPUT_FILE)


#import files
# Get the first row
first_row = train_data[20]
for i in range(20,25):
    first_row=train_data[i]
    #print(first_row['dialogue'])
    temp=first_row['dialogue']
    temp1=temp.split(":")
    print(type(temp1))
    for j in range(0,len(temp1)):
      text=temp1[j]
      if j%2==0:
        tts_object = gTTS(text=text, lang='en', tld='co.in',slow=False)
        audio_file = "output_dataset2_"+str(i)+str(j)+".mp3"
        tts_object.save(audio_file)
      else:
        tts_object = gTTS(text=text, lang='en', tld='co.uk',slow=False)
        audio_file = "output_dataset2_"+str(i)+str(j)+".mp3"
        tts_object.save(audio_file)
        #tts_object.save(audio_file)

      print(f"Audio successfully saved as {audio_file}")

# 5. Optional: Automatically play the audio file (Works on Windows/Mac/Linux)
# os.system(f"start {audio_file}")  # For Windows
# os.system(f"afplay {audio_file}") # For Mac

!zip -r processed_results.zip /content/


files.download('processed_results.zip')
