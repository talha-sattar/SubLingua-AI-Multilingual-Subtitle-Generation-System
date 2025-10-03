# Yam_Proc.py
import os
import json
import time
import subprocess
import logging
from pydub import AudioSegment
from pydub.utils import mediainfo
import shutil
# **** ADD THIS LINE ****
from typing import Dict, List, Optional # Import necessary types for hinting

# --- Configuration ---
# WARNING: Hardcoded paths. Consider using command-line arguments or a config file.
PROCESSING_DIR = "processing_steps"
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe" # IMPORTANT: Ensure .exe on Windows if providing full path. None uses PATH.

# Configure logging
LOG_FILE = "subtitle_processor.log"
# Clear log file at the start of the run
if os.path.exists(LOG_FILE):
    try:
        os.remove(LOG_FILE)
    except OSError as e:
        print(f"Warning: Could not clear previous log file: {e}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'), # Explicitly overwrite log
        logging.StreamHandler()
    ]
)

try:
    if FFMPEG_PATH and shutil.which(FFMPEG_PATH): 
        AudioSegment.converter = FFMPEG_PATH
        logging.info(f"Set pydub ffmpeg converter path to: {FFMPEG_PATH}")
       
        ffprobe_path = FFMPEG_PATH.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe") 

        if shutil.which(ffprobe_path):
             AudioSegment.ffprobe = ffprobe_path
             logging.info(f"Set pydub ffprobe path to: {ffprobe_path}")

        else:
             logging.warning(f"ffprobe not found at expected location: {ffprobe_path}. Pydub may still find it if in PATH.")

    elif FFMPEG_PATH:
         logging.warning(f"Provided FFMPEG_PATH '{FFMPEG_PATH}' not found or not executable. Pydub will try to find ffmpeg in PATH.")

    else:
        logging.info("FFMPEG_PATH not provided. Pydub will try to find ffmpeg in PATH.")
except Exception as e:
    logging.error(f"Error configuring pydub paths: {e}")
# --- End Configuration ---


class MediaProcessor:
    # Use class attributes for constants shared by all instances
    target_sr = 16000
    chunk_duration_ms = 975

    def __init__(self, processing_dir: str = PROCESSING_DIR, ffmpeg_path: Optional[str] = FFMPEG_PATH):
        self.supported_video = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
        self.supported_audio = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
        self.processing_dir = os.path.abspath(processing_dir) # Use absolute path for consistency
        self.ffmpeg_path = ffmpeg_path # Store the path if needed, though pydub config is primary

        try:
            os.makedirs(self.processing_dir, exist_ok=True)
        except OSError as e:
             logging.critical(f"Failed to create processing directory '{self.processing_dir}': {e}")
             raise # Stop if we cannot create the essential directory

        # Verify FFmpeg availability early if processing video (pydub handles audio conversion)
        # Note: shutil.which checks PATH. The pydub config above tries the direct path first.
        if self.ffmpeg_path and not shutil.which(self.ffmpeg_path):
             logging.warning(f"Provided FFmpeg path '{self.ffmpeg_path}' not found via shutil.which. "
                             "Video processing might fail if pydub also cannot find/use it.")
        elif not self.ffmpeg_path and not shutil.which("ffmpeg"): # Check path if no explicit path given
             logging.warning(f"FFmpeg command 'ffmpeg' not found in system PATH. "
                              "Video processing will likely fail.")
        logging.info(f"MediaProcessor initialized. Processing Dir: {self.processing_dir}")


    def process_media(self, input_path: str, overlap_ms: int = 0) -> Dict:
        """
        Process video/audio input, extract/convert audio, and generate fixed-duration chunks.
        Args:
            input_path: Path to input file (video/audio).
            overlap_ms: Overlap between chunks in milliseconds (default: 0).
                        Recommended: 487 (approx 50%) for better YAMNet coverage.
        Returns:
            Dictionary with processing results, chunk metadata, and errors.
        """
        start_time = time.time()
        results = {
            "input_path": input_path,
            "input_type": None,
            "processed_audio_path": None, # Renamed for clarity
            "duration_ms": 0,
            "chunks": [],
            "error": None,
            "warnings": []
        }

        try:
            # 1. Validate input
            self._validate_input(input_path)
            logging.info(f"Processing input: {input_path}")

            # 2. Clean processing directory (optional)
            self._clean_processing_dir()

            # 3. Determine file type and process/extract audio
            file_ext = os.path.splitext(input_path)[1].lower()
            # Create a unique base filename for processed files
            base_filename = os.path.splitext(os.path.basename(input_path))[0]
            timestamp = time.strftime("%Y%m%d%H%M%S")
            # Sanitize base_filename for safety
            safe_base = "".join(c if c.isalnum() or c in ['_','-'] else '_' for c in base_filename)
            processed_audio_filename = f"processed_{safe_base}_{timestamp}.wav"
            processed_audio_path = os.path.join(self.processing_dir, processed_audio_filename)

            if file_ext in self.supported_video:
                results["input_type"] = "video"
                results["processed_audio_path"] = self._process_video(input_path, processed_audio_path)
            elif file_ext in self.supported_audio:
                results["input_type"] = "audio"
                results["processed_audio_path"] = self._process_audio(input_path, processed_audio_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

            if not results["processed_audio_path"]:
                raise RuntimeError("Audio processing or extraction failed.")

            # 4. Verify the processed audio file
            audio_info = self._verify_audio_file(results["processed_audio_path"])
            results["duration_ms"] = audio_info["duration_ms"]

            # 5. Generate chunks from verified audio
            results["chunks"] = self._generate_chunks(
                results["processed_audio_path"],
                results["duration_ms"],
                overlap_ms
            )

            # 6. Save metadata about the processing run
            self._save_processing_metadata(results)

            logging.info(f"Successfully processed: {os.path.basename(input_path)}")

        except Exception as e:
            logging.error(f"Processing failed for {input_path}: {str(e)}", exc_info=True)
            results["error"] = str(e)
        finally:
             end_time = time.time()
             results["processing_time_s"] = round(end_time - start_time, 2)
             logging.info(f"Processing finished in {results['processing_time_s']:.2f} seconds.")
             return results # Always return the results dict

    def _clean_processing_dir(self):
        """Removes generated files (chunks, processed audio, metadata) from previous runs."""
        logging.info(f"Cleaning processing directory: {self.processing_dir}")
        if not os.path.isdir(self.processing_dir):
             logging.warning(f"Processing directory '{self.processing_dir}' does not exist, skipping cleaning.")
             return
        try:
            for filename in os.listdir(self.processing_dir):
                file_path = os.path.join(self.processing_dir, filename)
                try:
                    # Target specific generated files/patterns
                    if os.path.isfile(file_path) and \
                       (filename.startswith("chunk_") or
                        filename.startswith("processed_") or
                        filename in ["audio_verification.json", "processing_metadata.json", "chunk_metadata.json"]):
                           os.unlink(file_path)
                           logging.debug(f"Removed old file: {filename}")
                    # Avoid removing subdirectories or unrelated files unless intended
                except OSError as e:
                    # Log error but continue cleaning other files
                    logging.error(f'Failed to delete {file_path}. Reason: {e}')
        except Exception as e:
            logging.error(f"Error listing files during cleaning of directory '{self.processing_dir}': {e}")

    def _validate_input(self, input_path: str):
        """Validate input file existence and basic readability."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if not os.path.isfile(input_path):
             raise IsADirectoryError(f"Input path is a directory, not a file: {input_path}")
        if os.path.getsize(input_path) == 0:
             raise ValueError(f"Input file is empty: {input_path}")
        try:
             # Try getting basic info to check if readable by pydub/ffmpeg
             _ = mediainfo(input_path)
             logging.debug(f"Successfully read media info for validation: {input_path}")
        except Exception as e:
             # Catch potential errors from mediainfo (e.g., corrupted file)
             raise ValueError(f"Input file '{input_path}' seems corrupted or inaccessible: {e}")
        logging.debug(f"Validated input: {input_path}")

    def _process_video(self, video_path: str, output_wav_path: str) -> str:
        """Extract audio from video using FFmpeg (16 kHz, mono, WAV)."""
        logging.info(f"Extracting audio from video: {os.path.basename(video_path)}")
        # Use the configured ffmpeg path or let pydub/subprocess find it
        ffmpeg_executable = self.ffmpeg_path if self.ffmpeg_path and shutil.which(self.ffmpeg_path) else "ffmpeg"

        try:
            command = [
                ffmpeg_executable,
                 "-y", # Overwrite output without asking
                 "-i", video_path,
                 "-ac", "1",                  # Mono channel
                 "-ar", str(self.target_sr),  # Target sample rate (16kHz)
                 "-acodec", "pcm_s16le",      # Codec for 16-bit WAV
                 "-vn",                       # No video output
                 "-hide_banner",              # Suppress version info
                 "-loglevel", "warning",       # Show warnings and errors (or error for less noise)
                 output_wav_path
            ]
            logging.debug(f"Running FFmpeg command: {' '.join(command)}")
            process = subprocess.run(
                command,
                check=True, # Raise CalledProcessError if ffmpeg returns non-zero exit code
                capture_output=True,
                text=True,
                encoding='utf-8', # Be explicit about encoding
                errors='replace' # Handle potential encoding errors in output
            )
            # Log stderr which might contain warnings even on success
            if process.stderr:
                logging.warning(f"FFmpeg stderr output during audio extraction:\n{process.stderr.strip()}")

            # Check if output file was actually created and has size
            if not os.path.exists(output_wav_path) or os.path.getsize(output_wav_path) == 0:
                raise RuntimeError(f"Audio extraction seemed to succeed but output file is missing or empty: {output_wav_path}\nFFmpeg stderr: {process.stderr.strip()}")

            logging.info(f"Successfully extracted audio to: {output_wav_path}")
            return output_wav_path

        except FileNotFoundError:
             # This means the ffmpeg executable itself wasn't found
             error_msg = f"FFmpeg command '{ffmpeg_executable}' not found. Please install FFmpeg and ensure it's in your PATH or configure FFMPEG_PATH correctly."
             logging.critical(error_msg) # Use critical as it's a setup issue
             raise FileNotFoundError(error_msg)
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg failed with exit code {e.returncode} while extracting audio.\nCommand: {' '.join(e.cmd)}\nStderr: {e.stderr.strip()}\nStdout: {e.stdout.strip()}"
            logging.error(error_msg)
            raise RuntimeError("FFmpeg audio extraction failed.") from e
        except Exception as e:
            error_msg = f"An unexpected error occurred during FFmpeg execution: {str(e)}"
            logging.error(error_msg, exc_info=True)
            raise RuntimeError("Audio extraction failed.") from e

    def _process_audio(self, audio_path: str, output_wav_path: str) -> str:
        """Convert input audio to 16 kHz mono WAV using pydub."""
        logging.info(f"Processing audio file: {os.path.basename(audio_path)}")
        try:
            audio = AudioSegment.from_file(audio_path)
            logging.debug(f"Original audio info - SR: {audio.frame_rate}Hz, Channels: {audio.channels}, Len: {len(audio)}ms")

            # Resample and convert to mono
            audio = audio.set_frame_rate(self.target_sr).set_channels(1)
            logging.debug(f"Converted to SR: {audio.frame_rate}Hz, Channels: {audio.channels}")

            # Optional: Normalize loudness (can be disabled if original dynamics are preferred)
            # target_dbfs = -20.0
            # if -60 < audio.dBFS < target_dbfs - 1: # Avoid boosting silence or already loud audio
            #     change_in_dbfs = target_dbfs - audio.dBFS
            #     logging.info(f"Applying gain of {change_in_dbfs:.2f} dB (Original: {audio.dBFS:.2f} dBFS)")
            #     audio = audio.apply_gain(change_in_dbfs)
            # elif audio.dBFS >= target_dbfs -1:
            #      logging.info(f"Audio already loud enough ({audio.dBFS:.2f} dBFS), skipping normalization.")
            # else: # Very quiet
            #      logging.warning("Audio seems very quiet, skipping normalization.")

            # Export as 16-bit PCM WAV
            logging.debug(f"Exporting processed audio to {output_wav_path}")
            audio.export(output_wav_path, format="wav", parameters=["-acodec", "pcm_s16le"])
            logging.info(f"Processed audio saved to: {output_wav_path}")

            if not os.path.exists(output_wav_path) or os.path.getsize(output_wav_path) == 0:
                raise RuntimeError(f"Audio processing via pydub failed - output file missing or empty: {output_wav_path}")

            return output_wav_path

        except Exception as e:
            error_msg = f"Audio processing error for {audio_path} using pydub: {str(e)}"
            logging.error(error_msg, exc_info=True)
            # Provide hints if common errors occur
            if "ffmpeg" in str(e).lower() or "Couldn't find executable" in str(e):
                 error_msg += "\nHint: Ensure FFmpeg/FFprobe are installed and accessible via PATH or configured correctly."
            raise RuntimeError("Audio processing failed.") from e


    def _verify_audio_file(self, audio_path: str) -> Dict:
        """Verify processed audio meets 16 kHz, mono, and non-empty requirements."""
        logging.info(f"Verifying processed audio file: {os.path.basename(audio_path)}")
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Cannot verify - file not found: {audio_path}")
            if os.path.getsize(audio_path) == 0:
                 raise ValueError("Audio file is empty before reading info.")

            info = mediainfo(audio_path)
            # Safely extract and convert info
            sample_rate = float(info.get("sample_rate", 0))
            channels = int(info.get("channels", 0))
            duration_str = info.get("duration")
            duration_ms = float(duration_str) * 1000 if duration_str is not None else 0.0
            bits_per_sample = info.get("bits_per_sample", "unknown")
            file_size = os.path.getsize(audio_path) # Get size again after reading info

            verification = {
                "sample_rate": sample_rate,
                "channels": channels,
                "duration_ms": duration_ms,
                "bit_depth": bits_per_sample,
                "file_size": file_size
            }
            logging.debug(f"Raw mediainfo: {info}")
            logging.debug(f"Parsed verification data: {verification}")

            # Save verification data to JSON
            verification_path = os.path.join(self.processing_dir, "audio_verification.json")
            try:
                with open(verification_path, "w") as f:
                    json.dump(verification, f, indent=2)
            except Exception as json_e:
                logging.warning(f"Could not save verification json: {json_e}")

            # Validate required specs
            if not abs(sample_rate - self.target_sr) < 1: # Allow minor float inaccuracies
                raise ValueError(f"Invalid sample rate: {sample_rate} Hz (expected {self.target_sr} Hz)")
            if channels != 1:
                raise ValueError(f"Audio must be mono (found {channels} channels)")
            # Use a small threshold for minimum duration check
            min_sensible_duration_ms = 50
            if duration_ms < min_sensible_duration_ms :
                raise ValueError(f"Audio duration too short ({duration_ms:.0f} ms < {min_sensible_duration_ms} ms)")
            if file_size == 0: # Double check size
                raise ValueError("Audio file is empty after processing.")

            logging.info("Processed audio verification passed.")
            logging.info(f"  Duration: {duration_ms:.0f} ms")
            logging.info(f"  Sample Rate: {sample_rate/1000:.1f} kHz")
            logging.info(f"  Channels: {channels}")
            logging.info(f"  Bit Depth: {bits_per_sample}")
            return verification

        except Exception as e:
            logging.error(f"Audio verification failed for {audio_path}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Processed audio verification failed: {str(e)}") from e

    # Corrected type hint here
    def _generate_chunks(self, audio_path: str, duration_ms: float, overlap_ms: int) -> List[Dict]:
        """Generate fixed-duration audio chunks with padding."""
        logging.info(f"Generating {self.chunk_duration_ms}ms chunks with {overlap_ms}ms overlap...")
        if overlap_ms < 0:
             logging.warning(f"Negative overlap ({overlap_ms}ms) provided. Treating as 0ms.")
             overlap_ms = 0
        # Allow overlap equal to duration (will result in one chunk processed multiple times if step becomes 0)
        # Let step calculation handle it.
        if overlap_ms > self.chunk_duration_ms:
             logging.warning(f"Overlap ({overlap_ms}ms) > chunk duration ({self.chunk_duration_ms}ms). Step will be negative.")
             # Or raise ValueError("Overlap cannot exceed chunk duration.") if preferred

        try:
            audio = AudioSegment.from_file(audio_path) # Load the verified, processed audio
            chunks_meta: List[Dict] = [] # Ensure List[Dict] is used
            start_ms = 0
            # Step size can be 0 or negative if overlap >= duration
            step_ms = self.chunk_duration_ms - overlap_ms
            chunk_index = 0

            logging.debug(f"Chunking params: Total Duration={duration_ms:.0f}ms, ChunkDur={self.chunk_duration_ms}, Overlap={overlap_ms}, Step={step_ms}")

            # Check if total duration is sensible before starting loop
            if duration_ms < 1:
                 logging.warning("Total audio duration is near zero. No chunks will be generated.")
                 return []

            # Loop while the start point is less than the total duration
            # Use a safety counter to prevent potential infinite loops with weird steps
            max_iterations = int(duration_ms / abs(step_ms)) + 10 if step_ms != 0 else 2
            safety_counter = 0

            while start_ms < duration_ms and safety_counter < max_iterations:
                safety_counter += 1
                end_ms_ideal = start_ms + self.chunk_duration_ms
                actual_end_ms = min(end_ms_ideal, duration_ms) # Find the actual end within the audio file

                # Extract the slice - use integer indices
                chunk = audio[int(round(start_ms)):int(round(actual_end_ms))]

                # Minimum length check before padding - skip tiny fragments at the very end if desired
                min_fragment_len = 50 # ms - skip if slice is shorter than this
                if len(chunk) < min_fragment_len and start_ms > 0: # Don't skip the first chunk even if short
                     logging.debug(f"Skipping final audio fragment smaller than {min_fragment_len}ms (Actual len: {len(chunk)}ms)")
                     break # Stop chunking if only a tiny piece remains

                # Pad if the extracted chunk is shorter than the target duration
                if len(chunk) < self.chunk_duration_ms:
                    silence_needed = self.chunk_duration_ms - len(chunk)
                    # Use the audio object's frame rate for silence generation
                    try:
                        chunk += AudioSegment.silent(duration=silence_needed, frame_rate=audio.frame_rate)
                        logging.debug(f"Padded chunk {chunk_index} (start: {start_ms:.0f}ms) with {silence_needed}ms silence.")
                    except Exception as pad_e:
                        logging.error(f"Failed to pad chunk {chunk_index}: {pad_e}. Skipping chunk.")
                        start_ms += step_ms if step_ms != 0 else self.chunk_duration_ms # Advance start time
                        chunk_index += 1
                        continue

                # Define chunk metadata using ideal start/end for consistency in naming/processing
                chunk_start_time = int(round(start_ms)) # Use consistent integer ms timestamps
                chunk_end_time = chunk_start_time + self.chunk_duration_ms # Always use the fixed duration here

                chunk_filename = f"chunk_{chunk_index:05d}_{chunk_start_time:07d}_{chunk_end_time:07d}.wav"
                chunk_path = os.path.join(self.processing_dir, chunk_filename)

                try:
                    # Export the (potentially padded) chunk
                    chunk.export(chunk_path, format="wav", parameters=["-acodec", "pcm_s16le"])
                except Exception as export_e:
                    logging.error(f"Failed to export chunk {chunk_index} (Start: {start_ms:.0f}ms): {export_e}")
                    # Optionally skip this chunk and continue, or raise error
                    start_ms += step_ms if step_ms != 0 else self.chunk_duration_ms # Advance start time
                    chunk_index += 1
                    continue # Skip appending metadata for failed chunk

                chunks_meta.append({
                    "index": chunk_index,
                    "start": chunk_start_time, # Start time in original audio (ms)
                    "end": chunk_end_time,     # End time in original audio (ms) - based on fixed duration
                    "path": chunk_path,        # Path to the saved chunk file
                    "duration": self.chunk_duration_ms # Duration of the chunk file (should be fixed)
                })

                # Advance start time for the next chunk
                # Handle zero or negative step carefully
                if step_ms <= 0:
                     # If overlap is >= duration, only process the first chunk or advance by duration?
                     # Advancing by a fixed small amount might lead to excessive chunks.
                     # Let's advance by the chunk duration if step is not positive.
                     start_ms += self.chunk_duration_ms
                     if overlap_ms >= self.chunk_duration_ms:
                          logging.warning(f"Overlap >= chunk duration. Advancing by full chunk duration ({self.chunk_duration_ms}ms).")
                          # This prevents infinite loops but might miss overlaps if that was intended.
                else:
                     start_ms += step_ms

                chunk_index += 1

                # Safety break check
                if safety_counter >= max_iterations:
                     logging.error("Chunking loop reached maximum iterations. Breaking.")
                     break


            # --- Save Chunk Metadata ---
            chunk_meta_path = os.path.join(self.processing_dir, "chunk_metadata.json")
            try:
                with open(chunk_meta_path, 'w') as f:
                    json.dump(chunks_meta, f, indent=2)
                logging.info(f"Saved metadata for {len(chunks_meta)} chunks to {chunk_meta_path}")
            except Exception as e:
                logging.warning(f"Could not save chunk metadata: {e}")
            # ---

            logging.info(f"Successfully generated {len(chunks_meta)} audio chunks.")
            return chunks_meta

        except Exception as e:
            logging.error(f"Chunk generation failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Chunk generation failed: {str(e)}") from e


    def _save_processing_metadata(self, results: Dict):
        """Save overall processing metadata to a JSON file."""
        metadata_path = os.path.join(self.processing_dir, "processing_metadata.json")
        logging.debug(f"Saving processing metadata to {metadata_path}")
        # Create a copy to avoid modifying the original results dict during saving
        results_copy = results.copy()
        # Optionally remove full chunk paths from metadata if list is very long
        if 'chunks' in results_copy:
             # Just store number of chunks and maybe first/last chunk info in summary
             results_copy['chunks_summary'] = {
                 "count": len(results_copy.get("chunks", [])),
                 # Add other summary info if needed
             }
             del results_copy['chunks'] # Remove the full list from the saved metadata

        metadata = {
            "processor_version": "1.5", # Increment version if format changes
            "processing_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_file": os.path.basename(results.get("input_path", "N/A")),
            "results_summary": {
                 "input_type": results.get("input_type"),
                 "processed_audio_path": results.get("processed_audio_path"),
                 "duration_ms": results.get("duration_ms"),
                 "num_chunks": len(results.get("chunks", [])), # Get count from original results
                 "processing_time_s": results.get("processing_time_s"),
                 "error": results.get("error"),
                 "warnings": results.get("warnings")
            }
            # Keep full results if needed, or just summary
            # "full_results": results_copy # Contains the modified copy without full chunk paths
        }
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logging.info(f"Saved overall processing metadata to {metadata_path}")
        except Exception as e:
            logging.error(f"Failed to save overall processing metadata: {e}")


def main(path):

    INPUT_MEDIA_FILE = path
    ENABLE_OVERLAP = True # Set to True for overlapping chunks (recommended for YAMNet)
    OVERLAP_MS = 487 if ENABLE_OVERLAP else 0
    result = {
        'success': False,
        'error': None,
        'data': None,
        'warnings': []
    } 
    # ---

    print("=" * 50)
    print(" Media Processing Script (Yam_Proc) ")
    print("=" * 50)

    if not INPUT_MEDIA_FILE or not os.path.exists(INPUT_MEDIA_FILE):
        print(f"\nError: Input file not found or not specified: '{INPUT_MEDIA_FILE}'")
        logging.critical(f"Input file not found or not specified: '{INPUT_MEDIA_FILE}'")
        exit(1)

    print(f"\nProcessing input: {INPUT_MEDIA_FILE}")
    print(f"  Processing directory: {os.path.abspath(PROCESSING_DIR)}")
    print(f"  Target sample rate: {MediaProcessor.target_sr} Hz")
    print(f"  Chunk duration: {MediaProcessor.chunk_duration_ms} ms")
    print(f"  Chunk overlap: {OVERLAP_MS} ms")
    print(f"  Log file: {LOG_FILE}\n")

    logging.info("Script execution started.")
    processor = MediaProcessor(processing_dir=PROCESSING_DIR, ffmpeg_path=FFMPEG_PATH)
    result = processor.process_media(INPUT_MEDIA_FILE, overlap_ms=OVERLAP_MS)
    logging.info("Script execution finished.")

    
    print("-" * 50)
    print(" Processing Results")
    print("-" * 50)

    if not result.get("error"):
        print("Status: Processing Succeeded!")
        print(f"  Input Type: {result.get('input_type', 'N/A')}")
        print(f"  Processed Audio: {result.get('processed_audio_path', 'N/A')}")
        print(f"  Duration: {result.get('duration_ms', 0):.0f} ms")
        print(f"  Chunks Generated: {len(result.get('chunks', []))}")
        
        chunks_list = result.get("chunks", [])

        if chunks_list:
            print("\n  Example Chunks:")

            for chunk in chunks_list[:5]:

                print(f"    - Chunk {chunk.get('index', '?')}: {os.path.basename(chunk.get('path', 'N/A'))} "
                      f"({chunk.get('start')}ms - {chunk.get('end')}ms)")
                
            if len(chunks_list) > 5:
                print(f"    ... and {len(chunks_list) - 5} more chunks.")

        print(f"\n  Metadata saved in: {os.path.abspath(PROCESSING_DIR)}")
        print(f"  Processing Time: {result.get('processing_time_s', 0):.2f}s")

        


    else:
        print("Status: Processing Failed!")
        print(f"  Error: {result.get('error')}")
        # Warnings are not currently populated in the code, but could be added
        if result.get("warnings"):
             print("\n  Warnings:")

             for warn in result.get("warnings"): print(f"    - {warn}")
    print("-" * 50)

    result.update({
            'success': True,
            'data': {
                'processed_audio': result.get('processed_audio_path'),
                'duration_ms': result.get('duration_ms'),
                'chunks': result.get('chunks', []),
                'processing_time': result.get('processing_time_s')
            }
        })
        
    return result





if __name__ == "__main__":
    # --- User Configuration ---
    # WARNING: Hardcoded path. Use arguments or config in production.
    INPUT_MEDIA_FILE = r"C:\Users\tsatt\Downloads\Music\testing\nun.mp4"
    ENABLE_OVERLAP = True # Set to True for overlapping chunks (recommended for YAMNet)
    OVERLAP_MS = 487 if ENABLE_OVERLAP else 0 # Approx 50% overlap (975ms / 2 = 487.5ms)
    # ---

    print("=" * 50)
    print(" Media Processing Script (Yam_Proc) ")
    print("=" * 50)

    if not INPUT_MEDIA_FILE or not os.path.exists(INPUT_MEDIA_FILE):
        print(f"\nError: Input file not found or not specified: '{INPUT_MEDIA_FILE}'")
        logging.critical(f"Input file not found or not specified: '{INPUT_MEDIA_FILE}'")
        exit(1)

    print(f"\nProcessing input: {INPUT_MEDIA_FILE}")
    print(f"  Processing directory: {os.path.abspath(PROCESSING_DIR)}")
    print(f"  Target sample rate: {MediaProcessor.target_sr} Hz")
    print(f"  Chunk duration: {MediaProcessor.chunk_duration_ms} ms")
    print(f"  Chunk overlap: {OVERLAP_MS} ms")
    print(f"  Log file: {LOG_FILE}\n")

    logging.info("Script execution started.")
    processor = MediaProcessor(processing_dir=PROCESSING_DIR, ffmpeg_path=FFMPEG_PATH)
    result = processor.process_media(INPUT_MEDIA_FILE, overlap_ms=OVERLAP_MS)
    logging.info("Script execution finished.")

    print("-" * 50)
    print(" Processing Results")
    print("-" * 50)

    if not result.get("error"):
        print("Status: Processing Succeeded!")
        print(f"  Input Type: {result.get('input_type', 'N/A')}")
        print(f"  Processed Audio: {result.get('processed_audio_path', 'N/A')}")
        print(f"  Duration: {result.get('duration_ms', 0):.0f} ms")
        print(f"  Chunks Generated: {len(result.get('chunks', []))}")
        
        chunks_list = result.get("chunks", [])

        if chunks_list:
            print("\n  Example Chunks:")

            for chunk in chunks_list[:5]:
                print(f"    - Chunk {chunk.get('index', '?')}: {os.path.basename(chunk.get('path', 'N/A'))} "
                      f"({chunk.get('start')}ms - {chunk.get('end')}ms)")
                
            if len(chunks_list) > 5:
                print(f"    ... and {len(chunks_list) - 5} more chunks.")

        print(f"\n  Metadata saved in: {os.path.abspath(PROCESSING_DIR)}")
        print(f"  Processing Time: {result.get('processing_time_s', 0):.2f}s")

    else:
        print("Status: Processing Failed!")
        print(f"  Error: {result.get('error')}")
        
        if result.get("warnings"):
             print("\n  Warnings:")
             for warn in result.get("warnings"): print(f"    - {warn}")
    print("-" * 50)