!pip install -q kokoro>=0.9.2 soundfile
!apt-get -qq -y install espeak-ng > /dev/null 2>&1
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch
pipeline = KPipeline(lang_code='a')
text = '''
[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
'''


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
      text=temp1[j]
      if j%2==0:
        generator = pipeline(text, voice='af_heart')
        for i, (gs, ps, audio) in enumerate(generator):
          print(i, gs, ps)
          #display(Audio(data=audio, rate=24000, autoplay=i==0))
          sf.write("output4"+str(i)+str(j)+".wav", audio, 24000)
          print(f"Audio successfully saved as {audio_file}")
      else:
        generator = pipeline(text, voice='af_nicole')
        for i, (gs, ps, audio) in enumerate(generator):
          print(i, gs, ps)
          #display(Audio(data=audio, rate=24000, autoplay=i==0))
          sf.write("output4"+str(i)+str(j)+".wav", audio, 24000)

          print(f"Audio successfully saved as {audio_file}")

# 5. Optional: Automatically play the audio file (Works on Windows/Mac/Linux)
# os.system(f"start {audio_file}")  # For Windows
# os.system(f"afplay {audio_file}") # For Mac

!zip -r processed_results.zip /content/


files.download('processed_results.zip')
