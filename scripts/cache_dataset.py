# Caches the dataset into train, validation and test sets
import os
import sys

import torch 

# Timing
import time
start = time.time()

# Add the project path
cwd = os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Make the condor submission directory if it doesn't exist
os.makedirs(f"{cwd}/submission", exist_ok=True)
os.makedirs(f"{cwd}/submission/cache_logs", exist_ok=True)

from utils.preprocess import Preprocessor

#dataset_dir = "/vols/cms/pb4918/cmt/PreprocessRDF/dijet_2024/ttbar_inclusive/cat_base/prod_1707_filecopy"
#cache_dir = f"{cwd}/cachedir/ttbar"

#dataset_dir = "/vols/cms/pb4918/cmt/PreprocessRDF/dijet_2024/ttbar_inclusive/cat_base/prod_1807_filecopy"
#cache_dir = f"{cwd}/cachedir/ttbar"

dataset_dir = "/vols/cms/pb4918/StoreNTuple/L1Scouting/QCDProdwReco/QCD_15to7000/"
cache_dir = f"{cwd}/cachedir/qcd_15to7000"

# File list is all files ending in .root
dataset_files = [f"{dataset_dir}/{f}" for f in os.listdir(dataset_dir) if f.endswith(".root")]

# Train-valid-test split
train_frac = 0.6
valid_frac = 0.2
test_frac = 0.2

dataset_train_files = dataset_files[:int(len(dataset_files)*train_frac)]
dataset_valid_files = dataset_files[int(len(dataset_files)*train_frac):int(len(dataset_files)*(train_frac+valid_frac))]
dataset_test_files = dataset_files[int(len(dataset_files)*(train_frac+valid_frac)):]

print(f"Dataset {len(dataset_files)} files split into {len(dataset_train_files)} train, {len(dataset_valid_files)} valid, {len(dataset_test_files)} test")

# Make the cache directories
os.makedirs(f"{cache_dir}/train", exist_ok=True)
os.makedirs(f"{cache_dir}/validation", exist_ok=True)
os.makedirs(f"{cache_dir}/test", exist_ok=True)

print("\nCache directories created")

print("\nSetting up cache job")
# Write a list of arguments for condor jobs
# Args: INPUT_PATH OUTPUT_PATH
with open("submission/cache_file_args.txt", "w") as f:
    for file in dataset_train_files:
        filenum = (file.split("_")[-1]).split(".")[0]
        f.write(f"{file} {cache_dir}/train/file_{filenum}.pkl\n")
    for file in dataset_valid_files:
        filenum = (file.split("_")[-1]).split(".")[0]
        f.write(f"{file} {cache_dir}/validation/file_{filenum}.pkl\n")
    for file in dataset_test_files:
        filenum = (file.split("_")[-1]).split(".")[0]
        f.write(f"{file} {cache_dir}/test/file_{filenum}.pkl\n")

print("\nCache job arguments written to submission/cache_file_args.txt")

# Write the wrapper file
wrapper_file_content = f"""#!/bin/bash
cd {cwd}
eval "$(/vols/cms/pb4918/miniforge3/bin/conda shell.bash hook)"
conda activate ml_env
echo "ml env activated"
python3 scripts/cache_file.py $1 $2
"""

with open("submission/cache_file_wrapper.sh", "w") as f:
    f.write(wrapper_file_content)
os.system("chmod +x submission/cache_file_wrapper.sh")

print("\nCache job wrapper file created")

# Create the HTCondor job submission file
submit_file_content = f"""
universe = vanilla
executable = {cwd}/submission/cache_file_wrapper.sh
arguments = $(infile) $(outfile) 
output = {cwd}/submission/cache_logs/outputfile.$(CLUSTER)_$(PROCESS)
error = {cwd}/submission/cache_logs/errorfile.$(CLUSTER)_$(PROCESS)
log = {cwd}/submission/cache_logs/logfile.$(CLUSTER)_$(PROCESS)
request_cpus = 1
request_memory = 4GB
+MaxRuntime = 3599
queue infile, outfile from {cwd}/submission/cache_file_args.txt
"""

with open("submission/cache_files_job.sub", "w") as f:
    f.write(submit_file_content)
print("\nCache job submission file created")

# Delete existing log files
os.system(f"rm submission/cache_logs/*")

# Run the condor job
os.system(f"condor_submit submission/cache_files_job.sub")