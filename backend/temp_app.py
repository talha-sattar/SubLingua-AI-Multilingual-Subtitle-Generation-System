import os
import shutil
import re
import builtins
from flask import Flask, request, jsonify, send_from_directory, url_for, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS
import traceback
import logging
from flask import abort


from resources.preprocess import main as prp_main
from resources.model import main as mdl_main

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

# Configuration constants
UPLOAD_FOLDER = 'uploads'
DOWNLOAD_FOLDER = 'download'
OUTPUT_DIR_MODEL = os.path.abspath('output')
PROCESSING_FOLDER = 'processing_steps'
MAX_FILE_SIZE_MB = 2000
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Ensure directories exist
for folder in [UPLOAD_FOLDER, DOWNLOAD_FOLDER, OUTPUT_DIR_MODEL, PROCESSING_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config.update({
    'UPLOAD_FOLDER': UPLOAD_FOLDER,
    'DOWNLOAD_FOLDER': DOWNLOAD_FOLDER,
    'OUTPUT_DIR_MODEL': OUTPUT_DIR_MODEL,
    'PROCESSING_FOLDER': PROCESSING_FOLDER
})

@app.route('/api/upload', methods=['POST'])
def upload_and_process():
    """Handle file upload and processing pipeline"""
    try:
        # Clean temporary folders first
        clear_temporary_folders([UPLOAD_FOLDER, PROCESSING_FOLDER, OUTPUT_DIR_MODEL , DOWNLOAD_FOLDER])

        # Validate file upload
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

        # Get processing parameters
        processing_params = {
            'language': request.form.get('language', 'en').lower(),
            'genre': request.form.get('genre', 'General').strip(),
            'output_type': request.form.get('outputType', 'srt').lower(),
            'media_type': request.form.get('activeTab'),
            'delay': process_delay_input(request.form.get('delay', '0'))
        }

        # Log processing parameters
        logging.info(f"Processing parameters: {processing_params}")
        logging.info(f"File saved to: {upload_path}")

        # Execute preprocessing
        preprocess_results = prp_main(upload_path)

        if preprocess_results.get('error'):
        # If an error exists, return it in the response
            return jsonify({
            'status': 'failure',
            'error': preprocess_results['error']
        }), 500

        model_results = mdl_main(processing_params['language'] , processing_params['genre'], processing_params['output_type'] , processing_params['delay'])
            
        if model_results.get('error'):
        # If an error exists, return it in the response
            return jsonify({
            'status': 'failure',
            'error': preprocess_results['error']
        }), 500 

        print("---------- Success in model generation -----------")

        return jsonify({
        'status': 'success',
        'message': 'Subtitle generation completed successfully.'
    }), 200

    except Exception as e:
        # Enhanced error handling with nested try-except
        try:
            logging.error(f"Main processing error: {str(e)}", exc_info=True)
            error_msg = str(e)
        except Exception as log_error:
            error_msg = "An unexpected error occurred during error handling"
            logging.critical(f"Error handling failure: {str(log_error)}")
        
        return jsonify({'error': error_msg}), 500

def process_delay_input(delay_input: str) -> int:
    """Validate and process delay input"""
    try:
        clean_input = delay_input.lower().strip()
        if clean_input in ('none', ''):
            return 0
        return max(0, int(clean_input))
    except ValueError:
        logging.warning(f"Invalid delay value '{delay_input}' received. Using 0.")
        return 0

@app.route('/download/latest')
def download_latest():
    # Get the absolute path of the output directory
    abs_output_dir = os.path.abspath('output')
    logging.info(f"Absolute path of OUTPUT_DIR_MODEL: {abs_output_dir}")
    
    # Find files in the output directory
    files = [f for f in os.listdir(abs_output_dir) if os.path.isfile(os.path.join(abs_output_dir, f))]
    logging.info(f"Files found: {files}")
    if not files:
        return jsonify({"error": "No files found"}), 404
    
    # Construct the full file path
    file_path = os.path.join(abs_output_dir, files[0])
    logging.info(f"Serving file: {file_path}")
    
    # Verify the file exists before serving
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return jsonify({"error": "File not found"}), 404
    
    # Serve the file using send_file
    return send_file(file_path, as_attachment=True, download_name=files[0])

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
    app.run(debug=True)