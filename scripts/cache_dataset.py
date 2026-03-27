# Wrapper script to cache dataset
# Caches the dataset into train, validation and test sets
import os
import sys
import glob
import argparse

# Add the project path
cwd = os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Example: python3 scripts/cache_dataset.py --input root://gfe02.grid.hep.ph.ic.ac.uk/pnfs/hep.ph.ic.ac.uk/data/cms//store/user/ppradeep/L1Scouting/WtoMuNu-4Jets_TuneCP5_13p6TeV_madgraphMLM-pythia8/Summer24NanoV15WithL1/251217_045803/0000 --output cache/wjet_2024 --dryrun
parser = argparse.ArgumentParser(description='Preprocess Dataset')
parser.add_argument('--input', '-i', type=str, help='Input path')
parser.add_argument('--output', '-o', type=str, help='Output path')
parser.add_argument("--config", "-c", type=str, help="Config name")
parser.add_argument('--nfiles', '-n', default=-1, type=int, help='Number of files to process')
parser.add_argument('--dryrun', '-d', action="store_true", help='Dry run (do not submit jobs)')
args = parser.parse_args()

# Make the condor submission directory if it doesn't exist
parent_dir = f"{cwd}/condor_submission/cache"
os.makedirs(f"{parent_dir}", exist_ok=True)
os.makedirs(f"{parent_dir}/logs", exist_ok=True)

dataset_dir = args.input
cache_dir = args.output

if "root://" in dataset_dir:
    file_list = os.popen(f"gfal-ls {dataset_dir}").read().strip().split("\n")
else:    
    file_list = os.listdir(dataset_dir)

# File list is all files ending in .root
dataset_files = [f"{dataset_dir}/{f}" for f in file_list if f.endswith(".root")]

# Reduce filelist
n_files_to_process = args.nfiles
n_files_dataset = len(dataset_files)
if(args.nfiles > n_files_dataset):
    n_files_to_process = n_files_dataset
elif(args.nfiles < 0):
    n_files_to_process = n_files_dataset
else:
    n_files_to_process = args.nfiles

dataset_files = dataset_files[:n_files_to_process]

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
with open(f"{parent_dir}/args.txt", "w") as f:
    for file in dataset_train_files:
        filenum = (file.split("_")[-1]).split(".")[0]
        f.write(f"{file} {cache_dir}/train/file_{filenum}.pkl {args.config}\n")
    for file in dataset_valid_files:
        filenum = (file.split("_")[-1]).split(".")[0]
        f.write(f"{file} {cache_dir}/validation/file_{filenum}.pkl {args.config}\n")
    for file in dataset_test_files:
        filenum = (file.split("_")[-1]).split(".")[0]
        f.write(f"{file} {cache_dir}/test/file_{filenum}.pkl {args.config}\n")

print(f"Cache job arguments written to {parent_dir}/args.txt")

# Write the wrapper file
wrapper_file_content = f"""#!/bin/bash
cd {cwd}
source /cvmfs/cms.cern.ch/cmsset_default.sh
source /cvmfs/grid.cern.ch/alma9-ui-current/etc/profile.d/setup-alma9-test.sh
export X509_USER_PROXY={cwd}/condor_submission/cms.proxy

# Mamba init
export MAMBA_EXE='/home/hep/pb4918/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/home/hep/pb4918/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from micromamba activate
fi
unset __mamba_setup
eval "$(micromamba shell hook --shell bash)"
micromamba activate pt_regression_env
echo "pt_regression_env activated"

# Run script
python3 scripts/cache_file.py $1 $2 $3
"""

with open(f"{parent_dir}/cache_file_wrapper.sh", "w") as f:
    f.write(wrapper_file_content)
os.system(f"chmod +x {parent_dir}/cache_file_wrapper.sh")

print("Cache job wrapper file created")

# Create the HTCondor job submission file
submit_file_content = f"""
universe = vanilla
executable = {parent_dir}/cache_file_wrapper.sh
arguments = $(infile) $(outfile) 
output = {parent_dir}/logs/outputfile.$(CLUSTER)_$(PROCESS)
error = {parent_dir}/logs/errorfile.$(CLUSTER)_$(PROCESS)
log = {parent_dir}/logs/logfile.$(CLUSTER)_$(PROCESS)
request_cpus = 1
request_memory = 4GB
use_x509userproxy = true
+MaxRuntime = 3599
queue infile, outfile from {parent_dir}/args.txt
"""

with open(f"{parent_dir}/cache_files_job.sub", "w") as f:
    f.write(submit_file_content)
print("Cache job condor_submission file created")

# Run the condor job
if not args.dryrun:
    # Delete existing log files
    os.system(f"rm {parent_dir}/logs/*")
    os.system(f"condor_submit {parent_dir}/cache_files_job.sub")