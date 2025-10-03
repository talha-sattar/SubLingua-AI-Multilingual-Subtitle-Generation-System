import os
import json
import subprocess
from pydub import AudioSegment
from pydub.utils import mediainfo

def process_media(input_path: str, chunk_size: int = 1000, overlap: int = 500) -> dict:
    """
    Main media processing function.
    Handles both video and audio files, converts to standardized WAV format,
    and splits into chunks for temporal processing.
    """
    results = {
        "input_type": None,
        "audio_path": None,
        "chunks": [],
        "error": None
    }

    try:
        # Validate file type
        file_ext = os.path.splitext(input_path)[1].lower()
        supported_video = {".mp4", ".mkv", ".avi"}
        supported_audio = {".wav", ".mp3"}

        if file_ext in supported_video:
            results["input_type"] = "video"
            results["audio_path"] = extract_audio_from_video(input_path)
        elif file_ext in supported_audio:
            results["input_type"] = "audio"
            results["audio_path"] = process_audio_file(input_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        # Verify converted audio
        verify_audio_file(results["audio_path"])

        # Split audio into chunks
        results["chunks"] = split_audio_into_chunks(results["audio_path"], chunk_size, overlap)

        # Save processing metadata
        os.makedirs("processing_steps", exist_ok=True)
        with open("processing_steps/01_audio_processing.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"Successfully processed: {os.path.basename(input_path)}")
        return results

    except Exception as e:
        results["error"] = str(e)
        print(f"Processing failed: {str(e)}")
        return results

def extract_audio_from_video(video_path: str) -> str:
    """Extract audio from video and convert to 16kHz mono WAV using FFmpeg"""
    output_path = os.path.abspath("processing_steps/extracted_audio.wav")

    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Extract audio using FFmpeg
        command = [
            "ffmpeg", "-i", video_path,
            "-ac", "1", "-ar", "16000",  # Mono, 16kHz
            "-y", output_path  # Overwrite if exists
        ]
        subprocess.run(command, check=True)

        if not os.path.exists(output_path):
            raise RuntimeError("Audio extraction failed")

        return output_path

    except Exception as e:
        raise RuntimeError(f"Video processing error: {str(e)}")

def process_audio_file(audio_path: str) -> str:
    """Convert any audio file to standardized 16kHz mono WAV"""
    output_path = os.path.abspath("processing_steps/processed_audio.wav")

    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Get original format from extension
        orig_format = os.path.splitext(audio_path)[1][1:]  # Remove dot

        # Load and convert
        audio = AudioSegment.from_file(audio_path, format=orig_format)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")

        return output_path

    except Exception as e:
        raise RuntimeError(f"Audio conversion error: {str(e)}")

def verify_audio_file(audio_path: str) -> bool:
    """Validate audio meets specifications using Mediainfo"""
    try:
        # Get technical metadata
        info = mediainfo(audio_path)

        verification = {
            "sample_rate": float(info.get("sample_rate", 0)),
            "channels": int(info.get("channels", 0)),
            "duration": float(info.get("duration", 0)),
            "format": info.get("format_name", ""),
            "file_size": os.path.getsize(audio_path)
        }

        # Save verification data
        os.makedirs("processing_steps", exist_ok=True)
        with open("processing_steps/02_audio_verification.json", "w") as f:
            json.dump(verification, f, indent=2)

        # Validate specs
        assert verification["sample_rate"] == 16000, "Invalid sample rate"
        assert verification["channels"] == 1, "Must be mono audio"
        assert verification["duration"] > 0, "Empty audio file"

        print("Audio verification passed:")
        print(f"• Duration: {verification['duration']:.2f}s")
        print(f"• Sample Rate: {verification['sample_rate']/1000:.1f}kHz")
        print(f"• Channels: {verification['channels']}")

        return True

    except Exception as e:
        raise RuntimeError(f"Audio verification failed: {str(e)}")

def split_audio_into_chunks(audio_path: str, chunk_size: int, overlap: int) -> list:
    """
    Split audio into smaller chunks for temporal processing.
    Args:
        chunk_size: Length of each chunk in milliseconds.
        overlap: Overlap between chunks in milliseconds.
    Returns:
        List of chunk file paths.
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        chunks = []
        start = 0

        while start + chunk_size <= len(audio):
            end = start + chunk_size
            chunk = audio[start:end]
            chunk_path = os.path.abspath(f"processing_steps/chunk_{start}_{end}.wav")
            chunk.export(chunk_path, format="wav")
            chunks.append({
                "start": start,
                "end": end,
                "path": chunk_path
            })
            start += (chunk_size - overlap)

        return chunks

    except Exception as e:
        raise RuntimeError(f"Audio chunking failed: {str(e)}")

if __name__ == "__main__":
    # Example usage
    input_file = r"C:\Users\tsatt\Downloads\Music\testing\Explosions_30.wav"  # Updated path
    print("Starting audio processing...")
    result = process_media(input_path=input_file)

    if not result["error"]:
        print(f"Processed audio saved to: {result['audio_path']}")
        print(f"Generated {len(result['chunks'])} chunks:")
        for chunk in result["chunks"]:
            print(f"• {chunk['path']} ({chunk['start']}ms - {chunk['end']}ms)")
    else:
        print("Processing failed. Check error logs.")