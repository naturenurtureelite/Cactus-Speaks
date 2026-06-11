from pathlib import Path
from openai import OpenAI

# The client automatically picks up the OPENAI_API_KEY environment variable
client = OpenAI()

# Define where you want to save the output file
speech_file_path = Path(__file__).parent / "speech.mp3"

# Request speech synthesis with steerable instructions
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="coral",  # Other voices: alloy, ash, ballad, echo, fable, nova, onyx, sage, shimmer, verse
    input="Welcome",
    instructions="Speak in a cheerful, highly enthusiastic, and positive tone."
) as response:
    response.stream_to_file(speech_file_path)

print(f"Audio saved successfully to {speech_file_path}")





# The client automatically picks up the OPENAI_API_KEY environment variable
client = OpenAI()

# Define where you want to save the output file
speech_file_path = Path(__file__).parent / "speech2.mp3"

# Request speech synthesis with steerable instructions
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="alloy",  # Other voices: alloy, ash, ballad, echo, fable, nova, onyx, sage, shimmer, verse
    input="Welcome",
    instructions="Speak in a cheerful, highly enthusiastic, and positive tone."
) as response:
    response.stream_to_file(speech_file_path)

print(f"Audio saved successfully to {speech_file_path}")



