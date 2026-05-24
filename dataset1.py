import os
from pathlib import Path
from openai import OpenAI

# Initialize the client. It automatically picks up the OPENAI_API_KEY environment variable.
client = OpenAI()

# Define the output path for your generated audio file
output_audio_path = Path(__file__).parent / "output_speech.mp3"

try:
    print("Generating audio using gpt-4o-mini-tts...")
    
    # Call the speech creation endpoint
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="coral",        # Options: alloy, ash, ballad, coral, echo, fable, onyx, nova, sage, shimmer, verse
        input="Hello! This audio is being generated efficiently using the gpt-4o-mini text-to-speech model.",
        response_format="mp3" # Options: mp3, opus, aac, flac, wav, pcm
    )
    
    # Stream and write the raw binary audio response directly to a file
    response.stream_to_file(output_audio_path)
    
    print(f"Success! Audio successfully saved to: {output_audio_path}")

except Exception as e:
    print(f"An error occurred: {e}")
