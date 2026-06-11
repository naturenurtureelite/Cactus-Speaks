!pip install gTTS
from gtts import gTTS
import os

# 1. Define the text you want to convert
text_to_speech = "Hello! This is a quick demonstration of the Google Text to Speech library in Python."

# 2. Choose the language (English in this case)
# 'en' is the ISO code for English. You can use 'fr' for French, 'es' for Spanish, etc.
language = 'en'

# 3. Create the gTTS object
# 'slow=False' reads the text at normal speed. Set to True for slower speech.
tts_object = gTTS(text=text_to_speech, lang=language, tld='co.in',slow=False)

# 4. Save the converted audio to a file
audio_file = "output.mp3"
tts_object.save(audio_file)

print(f"Audio successfully saved as {audio_file}")

# 5. Optional: Automatically play the audio file (Works on Windows/Mac/Linux)
# os.system(f"start {audio_file}")  # For Windows
# os.system(f"afplay {audio_file}") # For Mac



from gtts import gTTS
import os

# 1. Define the text you want to convert
text_to_speech = "Hello! This is a quick demonstration of the Google Text to Speech library in Python."

# 2. Choose the language (English in this case)
# 'en' is the ISO code for English. You can use 'fr' for French, 'es' for Spanish, etc.
language = 'en'

# 3. Create the gTTS object
# 'slow=False' reads the text at normal speed. Set to True for slower speech.
tts_object = gTTS(text=text_to_speech, lang=language,tld='co.uk', slow=True)

# 4. Save the converted audio to a file
audio_file = "output.mp3"
tts_object.save(audio_file)

print(f"Audio successfully saved as {audio_file}")

# 5. Optional: Automatically play the audio file (Works on Windows/Mac/Linux)
# os.system(f"start {audio_file}")  # For Windows
# os.system(f"afplay {audio_file}") # For Mac
