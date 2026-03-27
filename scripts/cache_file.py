# Caches one file (Script to be run over a condor job)
import os 
import sys

# Imports from project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocess_object import Preprocessor

if len(sys.argv) != 4:
    print("Usage: cache_file.py <input_file> <output_file> <config_name>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]
config_name = sys.argv[3]

print(f"Args: {input_file} {output_file} {config_name}")

# Remove the output file if it exists
if os.path.exists(output_file):
    os.remove(output_file)
    print(f"Removed existing output file: {output_file}")

# Get the cache directory
cache_dir = os.path.dirname(output_file)

# Create a preprocessor object and cache
p = Preprocessor([input_file], cache_dir=cache_dir, config_name=config_name, use_existing_cache=False, batch_size=10000)
p.cache_file(input_file, output_file)
print("Cache file created")