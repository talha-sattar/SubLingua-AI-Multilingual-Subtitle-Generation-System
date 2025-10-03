import os
import shutil
import re
import builtins
from flask import Flask, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS
import traceback
import logging

from resources.preprocess import main as prp_main

# from resources.preprocess import MediaProcessor
# from resources.model import SubtitleGenerator
# import resources.model as model_module

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)



app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER =  'uploads'
DOWNLOAD_FOLDER =  'download'
OUTPUT_DIR_MODEL =  'output'
PROCESSING_FOLDER =  'processing_steps'

MAX_FILE_SIZE_MB = 2000

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_DIR_MODEL, exist_ok=True)
os.makedirs(PROCESSING_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['OUTPUT_DIR_MODEL'] = OUTPUT_DIR_MODEL
app.config['PROCESSING_FOLDER'] = PROCESSING_FOLDER



# --- Constrants ---

MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
FFMPEG_EXECUTABLE_PATH = os.environ.get("FFMPEG_PATH", None)


@app.route('/api/upload', methods=['POST'])
def upload_and_process():

    """Handle file upload and processing pipeline"""
    try:
        # Clean temporary folders
        clear_temporary_folders([UPLOAD_FOLDER, PROCESSING_FOLDER, OUTPUT_DIR_MODEL])

        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Validate file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        if file_size > MAX_FILE_SIZE_BYTES:
            return jsonify({'error': f'File size exceeds {MAX_FILE_SIZE_MB}MB limit'}), 413

        # Save uploaded file
        original_filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, original_filename)
        file.save(upload_path)

        # Get processing parameters with validation
        language = request.form.get('language', 'en').lower()
        genre = request.form.get('genre', 'General').strip()
        output_type = request.form.get('outputType', 'srt').lower()
        media_type = request.form.get('activeTab')
        delay_input = request.form.get('delay', '0').lower().strip()
        
        # Validate delay parameter
        
        try:
            if delay_input in ('none', ''):
                delay_input = 0
            else:
                delay_input = int(delay_input)

                if delay_input < 0:
                    logging.warning(f"Negative delay value {delay_input} received. Using 0.")
                    delay_input = 0

        except ValueError:
            logging.warning(f"Invalid delay value '{delay_input}' received. Using 0.")
            delay_input = 0

        print(f"File saved to: {upload_path}")
        print(f"Media type: {media_type}")
        print(f"Language: {language}")
        print(f"Genre: {genre}")
        print(f"Output Type: {output_type}")
        print(f"Delay: {delay_input}")

        # Prepare simulated inputs
        # genre_map = {
        #     'general': 'G',
        #     'none': 'N',
        #     'all': 'A'
        # }
        # genre_sim = genre_map.get(genre_input.lower(), 'G')
        # lang_sim = target_lang if target_lang != 'english' else ''
        # output_sim = output_type if output_type != 'srt' else ''
        # delay_sim = str(nv_delay_sec)

        # simulated_inputs = [genre_sim, lang_sim, output_sim, delay_sim]

        # Preprocessing
        preprocess_results = prp_main(upload_path)

        if preprocess_results:
            return jsonify({
                'status': 'success',
                'result': preprocess_results  # Or your desired result
            }), 200
        else:
            return jsonify({'error': 'Preprocessing failed'}), 500
        
        
    #     processor = MediaProcessor(processing_dir=PROCESSING_FOLDER, ffmpeg_path=FFMPEG_EXECUTABLE_PATH)
    #     process_result = processor.process_media(upload_path, overlap_ms=487)
        
    #     if not process_result.get("processed_audio_path"):
    #         return jsonify({'error': 'Audio processing failed'}), 500

    #     # Model processing with input simulation
    #     original_input = builtins.input
    #     input_simulator = InputSimulator(simulated_inputs)
    #     builtins.input = input_simulator.simulate_input

    #     try:
    #         generator = SubtitleGenerator()
    #         generator.generate_subtitles(
    #             audio_path=process_result['processed_audio_path'],
    #             transcription_language=transcription_lang
    #         )

    #         # Build output filename
    #         base_name = os.path.splitext(os.path.basename(process_result['processed_audio_path']))[0]
    #         match = re.match(r"processed_(.+?)_\d+", base_name)
    #         safe_name = match.group(1) if match else "output"
            
    #         output_filename = f"{safe_name}_{genre_input}Filter_{target_lang}_delay{nv_delay_sec}s.{output_type}"
    #         output_path = os.path.join(OUTPUT_DIR_MODEL, output_filename)

    #         if os.path.exists(output_path):
    #             final_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
    #             shutil.move(output_path, final_path)
    #             return jsonify({
    #                 'downloadUrl': url_for('download_file', filename=output_filename),
    #                 'filename': output_filename
    #             }), 200

    #         return jsonify({'error': 'Output file not generated'}), 500

    #     finally:
    #         builtins.input = original_input

    except Exception as e:
        logging.error(f"Processing error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    """Serve generated subtitle files"""
    if ".." in filename or filename.startswith("/"):
        return jsonify({"error": "Invalid filename"}), 403
        
    if not filename.lower().endswith(('.srt', '.vtt')):
        return jsonify({"error": "Invalid file type"}), 403

    try:
        return send_from_directory(DOWNLOAD_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

def clear_temporary_folders(folders: list):
    """Clean contents of specified folders"""
    for folder in folders:
        if os.path.exists(folder):
            for item in os.listdir(folder):
                path = os.path.join(folder, item)
                try:
                    if os.path.isfile(path) or os.path.islink(path):
                        os.unlink(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                except Exception as e:
                    logging.error(f"Error deleting {path}: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)