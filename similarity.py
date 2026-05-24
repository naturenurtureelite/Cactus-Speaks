import torch
import torchaudio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from speechbrain.inference.speaker import EncoderClassifier

# 1. Initialize SpeechBrain's pre-trained ECAPA-TDNN model
# This model will automatically download from HuggingFace on its first run.
print("Loading SpeechBrain speaker recognition model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    run_opts={"device": device}
)

def extract_speaker_embedding(audio_path):
    """
    Loads an audio file, normalizes it to the expected 16kHz sampling rate,
    and extracts a 192-dimensional speaker embedding vector.
    """
    # Load audio
    signal, fs = torchaudio.load(audio_path)
    
    # Pre-trained ECAPA-TDNN expects 16000Hz mono audio
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
        signal = resampler(signal)
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
        
    # Extract embedding
    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        # Flatten the output to a 1D vector: shape (192,)
        embedding_vector = embeddings.squeeze().cpu().numpy()
        
    return embedding_vector

def analyze_role_consistency(audio_files, role_label="Speaker"):
    """
    Extracts embeddings for a list of utterances, calculates their 
    pairwise cosine similarities, and checks consistency against an average profile.
    """
    print(f"\n--- Processing {role_label} Consistency ---")
    
    # Extract embeddings for all utterances
    embeddings = []
    for path in audio_files:
        try:
            emb = extract_speaker_embedding(path)
            embeddings.append(emb)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    if len(embeddings) < 2:
        print(f"Not enough valid utterances to analyze consistency for {role_label}.")
        return None

    embeddings = np.array(embeddings)
    
    # 1. Calculate Pairwise Cosine Similarity Matrix
    # Cosine similarity matrix gives the similarity of every file against every other file
    pairwise_sim = cosine_similarity(embeddings)
    
    # 2. Calculate Consistency against a global Profile
    # An 'anchor' voice profile is created by averaging all clean segments of that person
    profile_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
    consistency_scores = cosine_similarity(embeddings, profile_embedding).flatten()
    
    # Display Results
    print(f"Pairwise Similarity Matrix (Shape {pairwise_sim.shape}):")
    print(np.round(pairwise_sim, 3))
    
    print("\nConsistency Score against overall Profile (1.0 is perfect):")
    for i, score in enumerate(consistency_scores):
        print(f"  Utterance {i+1}: {score:.4f}")
        
    avg_consistency = np.mean(consistency_scores)
    print(f"** Average {role_label} Consistency Score: {avg_consistency:.4f} **")
    
    return pairwise_sim, avg_consistency

# =====================================================================
# Execution Block
# =====================================================================
if __name__ == "__main__":
    # Define paths to your segmented audio files (.wav format is preferred)
    patient_utterances = [
        "path/to/patient_chunk_1.wav",
        "path/to/patient_chunk_2.wav",
        "path/to/patient_chunk_3.wav"
    ]
    
    therapist_utterances = [
        "path/to/therapist_chunk_1.wav",
        "path/to/therapist_chunk_2.wav",
        "path/to/therapist_chunk_3.wav"
    ]
    
    # Mock analysis (Replace file arrays with real local paths to run)
    try:
        patient_sim, patient_avg = analyze_role_consistency(patient_utterances, role_label="Patient")
        therapist_sim, therapist_avg = analyze_role_consistency(therapist_utterances, role_label="Therapist")
    except FileNotFoundError:
        print("\n[Notice] Script initialized perfectly! To see real outputs, update the audio lists above with actual file paths.")
