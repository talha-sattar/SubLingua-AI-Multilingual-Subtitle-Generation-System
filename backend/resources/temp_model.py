import os
import numpy as np
import torch
import torchaudio
import librosa
import torch.nn as nn
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
from openai import OpenAI
import json
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# Custom Model Architecture
class AudioCNN(nn.Module):
    """CNN for spectrogram classification."""
    def __init__(self, num_classes, max_segments=10):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(max_segments, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * (128 // 4) * (501 // 4), 256)  # For 5s spectrograms
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Audio Processing Functions
def load_and_resample_audio(filepath, target_sr=16000):
    """Load and resample audio file to target sample rate."""
    try:
        audio, sr = librosa.load(filepath, sr=None, mono=True)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        if len(audio) < 100 or np.isnan(audio).any() or np.isinf(audio).any():
            print(f"Invalid audio in {filepath}: too short or contains NaN/Inf")
            return None
        return audio
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def segment_audio(audio, sr=16000, segment_length_sec=5.0):
    """Segment audio into 5-second segments, padding if needed."""
    segment_samples = int(segment_length_sec * sr)
    segments = []
    if len(audio) < segment_samples:
        audio = np.pad(audio, (0, segment_samples - len(audio)), mode='constant')
    segments.append(audio[:segment_samples])
    return segments

def extract_spectrogram(segment, sr=16000, n_mels=128):
    """Extract log-mel spectrogram with 501 time steps for 5s."""
    try:
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=512,
            win_length=int(0.025 * sr),
            hop_length=int(0.010 * sr),  # ~501 time steps for 5s
            n_mels=n_mels
        )
        mel_spec = transform(torch.from_numpy(segment).float())
        mel_spec = torch.log(mel_spec + 1e-10)
        # Ensure 501 time steps
        target_time_steps = 501
        if mel_spec.shape[1] < target_time_steps:
            mel_spec = torch.nn.functional.pad(mel_spec, (0, target_time_steps - mel_spec.shape[1]))
        elif mel_spec.shape[1] > target_time_steps:
            mel_spec = mel_spec[:, :target_time_steps]
        if torch.isnan(mel_spec).any() or torch.isinf(mel_spec).any():
            print(f"Invalid spectrogram generated")
            return torch.zeros((n_mels, target_time_steps))
        return mel_spec
    except Exception as e:
        print(f"Error extracting spectrogram: {e}")
        return torch.zeros((n_mels, 501))

def prepare_input(filepath, max_segments=10):
    """Prepare spectrogram input for custom model."""
    audio = load_and_resample_audio(filepath)
    if audio is None:
        return torch.zeros((max_segments, 128, 501))
    
    segments = segment_audio(audio)
    spectrograms = []
    for segment in segments:
        spec = extract_spectrogram(segment)
        spectrograms.append(spec)
    
    if not spectrograms:
        print(f"No valid segments for {filepath}")
        return torch.zeros((max_segments, 128, 501))
    
    if len(spectrograms) < max_segments:
        padding = [torch.zeros_like(spectrograms[0])] * (max_segments - len(spectrograms))
        spectrograms.extend(padding)
    else:
        spectrograms = spectrograms[:max_segments]
    
    spectrograms = torch.stack(spectrograms)
    return spectrograms

# VGGish Processing
def load_vggish_model(model_url):
    """Load VGGish model from TensorFlow Hub."""
    return hub.load(model_url)

def get_vggish_embeddings(audio_path, vggish):
    """Extract VGGish embeddings from an audio chunk."""
    try:
        audio = tf.io.read_file(audio_path)
        waveform, sample_rate = tf.audio.decode_wav(audio, desired_channels=1)
        waveform = tf.squeeze(waveform, axis=-1)
        
        if sample_rate != 16000:
            raise ValueError(f"Sample rate must be 16kHz. Found {sample_rate}Hz.")
        
        embeddings = vggish(waveform)
        return embeddings.numpy()
    except Exception as e:
        print(f"Error processing {audio_path} with VGGish: {e}")
        return None

# Custom Model Prediction
def detect_events(audio_chunk_path, model, device, class_names):
    """Detect audio events in a chunk using custom model."""
    model.eval()
    input_tensor = prepare_input(audio_chunk_path)
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        predicted_idx = torch.argmax(output, dim=1).item()
    
    predicted_class = class_names[predicted_idx]
    confidence = probabilities[predicted_idx]
    return predicted_class, confidence, probabilities

# Process Chunks
def process_audio_chunks(chunks_dir, model, device, class_names, vggish):
    """Process all audio chunks with VGGish and custom model."""
    results = []
    
    for chunk in sorted(os.listdir(chunks_dir)):
        if chunk.startswith("chunk_") and chunk.endswith(".wav"):
            chunk_path = os.path.join(chunks_dir, chunk)
            start_time = int(chunk.split("_")[1])
            end_time = int(chunk.split("_")[2].split(".")[0])
            
            # Get VGGish embeddings
            embeddings = get_vggish_embeddings(chunk_path, vggish)
            if embeddings is None:
                print(f"Skipping {chunk_path} due to VGGish processing error")
                continue
            
            # Get custom model prediction
            predicted_class, confidence, probabilities = detect_events(chunk_path, model, device, class_names)
            
            results.append({
                "start_time": start_time,
                "end_time": end_time,
                "label": predicted_class,
                "confidence": float(confidence),
                "embeddings": embeddings.tolist()
            })
    
    return results

# Translation
def translate_labels(results, target_language, api_key):
    """Translate labels to the target language using OpenAI API."""
    client = OpenAI(api_key=api_key)
    translated_results = []
    
    for result in results:
        try:
            prompt = f"Translate the following text to {target_language} while keeping it concise: {result['label']}"
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50
            )
            translated_label = response.choices[0].message.content.strip()
            translated_results.append({
                "start_time": result["start_time"],
                "end_time": result["end_time"],
                "label": translated_label,
                "confidence": result["confidence"],
                "embeddings": result["embeddings"]
            })
        except Exception as e:
            print(f"Error translating '{result['label']}': {str(e)}. Keeping original label.")
            translated_results.append(result)
    
    return translated_results

# SRT/VTT Formatting
def format_time_srt(milliseconds):
    """Convert milliseconds to SRT timestamp format (HH:MM:SS,mmm)."""
    seconds = milliseconds // 1000
    milliseconds = milliseconds % 1000
    minutes = seconds // 60
    seconds = seconds % 60
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def format_time_vtt(milliseconds):
    """Convert milliseconds to VTT timestamp format (HH:MM:SS.mmm)."""
    seconds = milliseconds // 1000
    milliseconds = milliseconds % 1000
    minutes = seconds // 60
    seconds = seconds % 60
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def save_srt(results, output_path):
    """Save results as an SRT file."""
    with open(output_path, "w") as f:
        for i, result in enumerate(results):
            start_time = format_time_srt(result["start_time"])
            end_time = format_time_srt(result["end_time"])
            label = result["label"]
            f.write(f"{i+1}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{label}\n\n")

def save_vtt(results, output_path):
    """Save results as a VTT file."""
    with open(output_path, "w") as f:
        f.write("WEBVTT\n\n")
        for i, result in enumerate(results):
            start_time = format_time_vtt(result["start_time"])
            end_time = format_time_vtt(result["end_time"])
            label = result["label"]
            f.write(f"{i+1}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{label}\n\n")

# Main Function
def main(target_language, output_format , filename , genre):
    # Configuration
    VGGISH_MODEL_URL = 'https://tfhub.dev/google/vggish/1'
    MODEL_CONFIGS = {
        "Action_War_Sounds": {
            "model_path": r"C:\Users\tsatt\Downloads\Music\Models\Action_War_Sounds_Models\Action_War_Sounds_WAV_best_model.pth",
            "classes": [
                'Body Impact', 'Bullets ricocheting', 'Car crashing', 'Explosions',
                'Glass shattering', 'Gunfire', 'Helicopter Blades', 'Rifle Loading',
                'Rocket Launcher Firing', 'Shouting', 'Tanks Rolling'
            ]
        },
        "General_Sounds": {
            "model_path": r"C:\Users\tsatt\Downloads\Music\Models\General_Sounds_Models\General_Sounds_WAV_best_model.pth",
            "classes": [
                'birds_chirping', 'clock_ticking', 'coughing', 'door_creaking',
                'footsteps', 'glass_breaking', 'heartbeat', 'heavy_breathing',
                'knocking', 'laughter', 'phone_ringing', 'rustling',
                'sighing', 'tapping', 'typing_on_keyboard', 'whispering',
                'wind_blowing'
            ]
        },
        "Horror_Suspense_Sounds": {
            "model_path": r"C:\Users\tsatt\Downloads\Music\Models\Horror_Suspense_Sounds_Models\best_model.pth",
            "classes": [
                'Chains rattling', 'Door Creaking', 'HeartBeat', 'Sudden scream',
                'Creaking floors', 'Eerie silence', 'Loud thud', 'Wind howling',
                'Distant Moaning', 'Ghostly whispering', 'Low berathing'
            ]
        },
        "Romantic_Emotional_Sounds": {
            "model_path": r"C:\Users\tsatt\Downloads\Music\Models\Romantic_Emotional_Sounds_Models\Romantic_Emotional_Sounds_WAV_best_model.pth",
            "classes": [
                'Fireplace crackling', 'HeartBeat', 'Laughter', 'sighing',
                'Soft Whispers', 'Glass Clinking', 'Kiss Sound', 'Raindrops Falling',
                'Soft Music'
            ]
        },
        "Dramatic_Suspenseful_Sounds": {
            "model_path": r"C:\Users\tsatt\Downloads\Music\Models\Dramatic_Suspenseful_Sounds_Models\Dramatic_Suspenseful_Sounds_WAV_best_model.pth",
            "classes": [
            'Chains rattling', 'Elevator ding', 'Muffled voices', 'Thunder',
            'Water dripping', 'Distant screams', 'Fire crackling', 'Shuffling footsteps',
            'Tires screeching', 'Wind howling'
            ]
        }
    }

    CHUNKS_DIR = "processing_steps"
    OUTPUT_BASE_PATH = f"download/{filename}"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load VGGish model
    print("Loading VGGish model...")
    vggish = load_vggish_model(VGGISH_MODEL_URL)
    
    # Model selection
    # print("\nAvailable models:")
    # for model_name in MODEL_CONFIGS.keys():
    #     print(f"- {model_name}")
    # while True:
    #     selected_model = input("Select a model: ").strip()
    #     if selected_model in MODEL_CONFIGS:
    #         break
    #     print("Invalid model. Please choose from the list.")


    if genre == 'general':
        selected_model = 'General_Sounds'

    elif genre == 'action':
        selected_model = "Action_War_Sounds"

    elif genre == 'romance':
        selected_model = "Romantic_Emotional_Sounds"

    elif genre == 'dramatic':
        selected_model = "Dramatic_Suspenseful_Sounds"

    elif genre == 'horror':
        selected_model = "Horror_Suspense_Sounds"
    
    model_config = MODEL_CONFIGS[selected_model]
    model_path = model_config["model_path"]
    class_names = model_config["classes"]
    
    # Load custom model
    print(f"Loading {selected_model} model...")
    model = AudioCNN(num_classes=len(class_names))
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process audio chunks
    print("Processing audio chunks...")
    results = process_audio_chunks(CHUNKS_DIR, model, device, class_names, vggish)
    
    # Save raw results
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    with open(os.path.join(CHUNKS_DIR, "03_chunk_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Display results in English
    print("\nResults in English:")
    for result in results:
        print(f"{result['start_time']}ms - {result['end_time']}ms: "
              f"{result['label']} (Confidence: {result['confidence']:.2f})")
    
    # Get output language
    # print("\nAvailable languages: English (default), Spanish, French, German, etc.")
    # target_language = input("Enter target language (or press Enter for English): ").strip().lower()
    if target_language == "" or target_language == "english":
        final_results = results
    else:
        print(f"Translating to {target_language}...")
        final_results = translate_labels(results, target_language, OPENAI_API_KEY)
    
    # Display translated results
    print("\nFinal results:")
    for result in final_results:
        print(f"{result['start_time']}ms - {result['end_time']}ms: "
              f"{result['label']} (Confidence: {result['confidence']:.2f})")
    
    # Get output format
    # while True:
    #     output_format = input("Enter output format (srt/vtt): ").lower()
    #     if output_format in ['srt', 'vtt']:
    #         break
    #     print("Invalid format. Please enter 'srt' or 'vtt'.")
    
    # Save results
    output_path = f"{OUTPUT_BASE_PATH}.{output_format}"
    if output_format == 'srt':
        save_srt(final_results, output_path)
    else:
        save_vtt(final_results, output_path)
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()