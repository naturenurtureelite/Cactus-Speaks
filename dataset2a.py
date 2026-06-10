from google.colab import files


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
for i in range(0,10):
    first_row=train_data[i]
    #print(first_row['dialogue'])
    temp=first_row['dialogue']
    temp1=temp.split(":")
    print(type(temp1))
    for j in range(0,len(temp1)):
      TEXT = temp1[j]
      if j%2==0:
        VOICE = "en-US-ChristopherNeural"
      else:
        VOICE="en-HK-YanNeural"
      OUTPUT_FILE = "test"+str(i)+str(j)+".mp3"
      RATE = "+10%"
      await main()
      #files.download("test"+str(i)+str(j)+".mp3")

!zip -r processed_results.zip /content/


files.download('processed_results.zip')
