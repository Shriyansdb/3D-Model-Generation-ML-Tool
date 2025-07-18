# Create test_generate.py
from generator import generate_3d_asset

result = generate_3d_asset("a simple cube", "outputs")
print(f"Generated assets: {result}")