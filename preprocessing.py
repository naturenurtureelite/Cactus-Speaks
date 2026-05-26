import re

def preprocess_utterance(text: str, pause_type: str = "standard") -> str:
    """
    Preprocesses raw text for speech synthesis/recognition based on four rules:
    1. Uppercase all characters.
    2. Remove intermediate punctuation.
    3. Terminate with a period or question mark.
    4. Replace spaces with custom pause duration tokens.
    
    Parameters:
    - text (str): The raw input utterance.
    - pause_type (str): The default pause type for standard spaces 
                        ('standard', 'slurred', 'short', 'long').
    """
    # Define the four word separators
    pause_tokens = {
        "slurred": "~",
        "standard": " ",
        "short": "/",
        "long": "%"
    }
    
    # Grab the target separator token
    sep = pause_tokens.get(pause_type, " ")
    
    # 1. Clean up tracking/whisper spaces and force uppercase (Rule 1)
    text = text.strip().upper()
    
    if not text:
        return ""
    
    # 2. Determine and isolate the final termination punctuation (Rule 3)
    # Checks if it ends with a question mark; defaults to a period otherwise.
    end_char = "?" if text.endswith("?") else "."
    
    # Strip any trailing punctuation so we can process the body uniformly
    text = re.sub(r'[.,?!;:\"\']+$', '', text)
    
    # 3. Remove all intermediate punctuation marks (Rule 2)
    # Retains alphanumeric characters and spaces for token replacement
    text = re.sub(r'[^\w\s]', '', text)
    
    # Collapse multiple consecutive spaces down to a single space
    text = re.sub(r'\s+', ' ', text)
    
    # 4. Replace spaces between words with the acoustic separator (Rule 4)
    # We use a lookahead and lookbehind to only replace spaces strictly between words
    processed_text = re.sub(r'(?<=\S)\s(?=\S)', sep, text)
    
    # Append the termination character (Rule 3)
    processed_text += end_char
    
    return processed_text

# ==========================================
# Example Usage & Verification
# ==========================================
if __name__ == "__main__":
    # Test 1: Standard space behavior
    raw_sentence = "Either way, you should shoot very slowly,"
    print(f"Raw:     {raw_sentence}")
    print(f"Default: {preprocess_utterance(raw_sentence, 'standard')}\n")
    
    # Test 2: Simulating custom manual pause injection logic
    # If your text already has explicit pause marks before passing to the script,
    # you can bypass the uniform space replacement or handle them token by token.
    print("--- Advanced Text-to-Acoustic Mapping Example ---")
    
    # Let's say you parse a sentence where pauses are explicitly marked in brackets:
    sentence_with_markers = "Either way [LONG] you should shoot [SHORT] very slowly"
    
    # Convert placeholders to temporary unique strings to protect them from punctuation stripping
    step1 = sentence_with_markers.upper()
    step1 = step1.replace("[LONG]", "===LONG===").replace("[SHORT]", "===SHORT===")
    step1 = re.sub(r'[^\w\s\-=]', '', step1) # Strip punctuation but keep our marker flags
    
    # Replace spaces with standard or custom tokens
    words = step1.split()
    final_tokens = []
    
    for i, word in enumerate(words):
        if word == "===LONG===":
            # Change the previous transition or current slot to %
            if final_tokens: final_tokens[-1] = final_tokens[-1].rstrip() + "%"
        elif word == "===SHORT===":
            if final_tokens: final_tokens[-1] = final_tokens[-1].rstrip() + "/"
        else:
            final_tokens.append(word + " ")
            
    final_str = "".join(final_tokens).strip() + "."
    # Fix any dangling spaces before punctuation or clean up standard spacing
    final_str = re.sub(r'\s+', ' ', final_str)
    
    print(f"Input:  {sentence_with_markers}")
    print(f"Output: {final_str}")
