@echo off
echo Setting up 3D Model Generator...

REM Create virtual environment
python -m venv venv
call venv\Scripts\activate

REM Install dependencies
pip install -r requirements.txt

REM Initialize models
python -c "import generator; generator.init_models()"

REM Create test asset
python -c "import generator; generator.generate_test_asset()"

echo Setup completed! Run 'python app.py' to start the server.
pause