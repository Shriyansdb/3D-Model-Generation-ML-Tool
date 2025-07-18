from flask import Flask, request, send_file, jsonify, render_template, send_from_directory
import generator
import os
import glob
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("3DGeneratorAPI")

app = Flask(__name__)
OUTPUT_DIR = "outputs"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate3d')
def generate_3d_endpoint():
    prompt = request.args.get('prompt', '')
    format_type = request.args.get('format', 'glb')  # Default to GLB
    
    if not prompt:
        return jsonify(error="Missing 'prompt' parameter"), 400
    
    try:
        # Generate assets
        logger.info(f"Received generation request: {prompt}")
        result = generator.generate_3d_asset(prompt, OUTPUT_DIR)
        
        if not result.get('success', False):
            error = result.get('error', 'Unknown generation error')
            logger.error(f"Generation failed: {error}")
            return jsonify(error=error), 500
        
        # Return requested format
        if format_type.lower() == 'obj' and result.get('obj'):
            return send_file(result['obj'], as_attachment=True, download_name=f"{prompt[:20]}.obj")
        elif result.get('glb'):
            return send_file(result['glb'], as_attachment=True, download_name=f"{prompt[:20]}.glb")
        else:
            return jsonify(error="No valid file generated"), 500
    
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return jsonify(error=str(e)), 500

@app.route('/preview')
def get_preview():
    prompt = request.args.get('prompt', '')
    if not prompt:
        return jsonify(error="Missing 'prompt' parameter"), 400
    
    clean_prompt = "".join(x for x in prompt[:50] if x.isalnum() or x in " _-")
    preview_path = f"{OUTPUT_DIR}/previews/{clean_prompt}_*.png"
    previews = glob.glob(preview_path)
    
    if previews:
        # Get most recent preview
        latest_preview = max(previews, key=os.path.getctime)
        return send_file(latest_preview, mimetype='image/png')
    return jsonify(error="Preview not found"), 404

@app.route('/status')
def status():
    def get_dir_size(path):
        return sum(os.path.getsize(f) for f in glob.glob(path + '/**', recursive=True) if os.path.isfile(f))
    
    try:
        # Test if generator is working
        generator.init_models()
        status_msg = "operational"
    except:
        status_msg = "degraded"
    
    return jsonify(
        status=status_msg,
        model="Shap-E-text300M",
        format_supported=["glb", "obj"],
        storage_path=os.path.abspath(OUTPUT_DIR),
        storage_usage=f"{get_dir_size(OUTPUT_DIR)/1024/1024:.2f} MB"
    )

@app.route('/test')
def test_endpoint():
    try:
        result = generator.generate_test_asset()
        if result.get('success', False):
            return jsonify({
                "status": "success",
                "files": result
            })
        else:
            return jsonify(error=result.get('error', 'Test failed')), 500
    except Exception as e:
        return jsonify(error=str(e)), 500

# Serve static files from outputs directory
@app.route('/outputs/<path:filename>')
def output_files(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    # Create output directories if missing
    for d in [OUTPUT_DIR, f"{OUTPUT_DIR}/objs", f"{OUTPUT_DIR}/glbs", f"{OUTPUT_DIR}/previews"]:
        os.makedirs(d, exist_ok=True)
    
    # Run self-test on startup
    try:
        test_result = generator.generate_test_asset()
        if test_result.get('success'):
            logger.info(f"Self-test completed successfully: {test_result}")
        else:
            logger.error(f"Startup self-test failed: {test_result.get('error')}")
    except Exception as e:
        logger.error(f"Startup self-test failed: {str(e)}")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)