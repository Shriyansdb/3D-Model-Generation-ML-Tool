Text-to-3D Generator using Shap-E
Generate 3D models from text prompts with a simple web API, powered by OpenAI’s Shap-E. This project offers straightforward setup, an easy-to-use API, CLI/test scripts, and outputs compatible with standard 3D tools.

Features :
Text-to-3D conversion: Generate .OBJ, .GLB, and preview .PNG files from English prompts.

REST API: Web interface to request file generation, get server status, and retrieve previews.

Robust startup tests: Runs a self-test and can check health/status anytime.

Organized outputs: Assets and previews saved under outputs/ with auto-generated filenames.

Quickstart
1. Clone the Repository
bash

git clone https://github.com/yourusername/text-to-3d-generator.git
cd text-to-3d-generator

2. Install & Configure

Windows (automated)
bash

setup.bat


Manual Setup (any OS)
bash

python -m venv venv

# Activate the environment:

#   venv\Scripts\activate   (Windows)
#   source venv/bin/activate (Linux/macOS)

pip install -r requirements.txt

python -c "import generator; generator.init_models()"
python -c "import generator; generator.generate_test_asset()"

Usage
API
Run the server:

bash
python app.py
Server is now running at http://localhost:5000.


Directory Structure:
app.py — Flask web API server

generator.py — Model load/generation logic

utils.py — Asset and preview file helpers

requirements.txt — Dependencies

setup.bat — Windows installation helper

test_generator.py — Simple demo/test driver

outputs/ — Generated assets and previews