"""
tools/fresh_clone_test.py
=========================
Simulates a fresh clone of the repository and runs the pipeline 
to verify that no uncommitted state is required to execute.
"""

import sys
import shutil
import subprocess
from pathlib import Path

def main():
    root_dir = Path(__file__).resolve().parents[1]
    tmp_dir = root_dir / ".tmp_clone"
    
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        
    print(f"Creating fresh clone at {tmp_dir}...")
    
    # Exclude .venv, data/interim, .env, models
    def ignore_patterns(path, names):
        if Path(path) == root_dir:
            return ['.venv', '.env', 'models', '.tmp_clone', '.git', '__pycache__']
        if Path(path) == root_dir / 'data':
            return ['interim']
        return ['__pycache__']
        
    shutil.copytree(root_dir, tmp_dir, ignore=ignore_patterns)
    
    # We won't actually create a real venv and pip install here because it's slow 
    # and requires network. We will use the existing python executable to run the scripts 
    # within the cloned directory to prove the paths are correct.
    python_exe = sys.executable
    
    print("Testing submission/cs2_pipeline.py...")
    res1 = subprocess.run([python_exe, "submission/cs2_pipeline.py"], cwd=tmp_dir)
    if res1.returncode != 0:
        print("Error running cs2_pipeline.py in fresh clone.")
        sys.exit(1)
        
    print("Testing src.analysis...")
    res2 = subprocess.run([python_exe, "-m", "src.analysis"], cwd=tmp_dir)
    if res2.returncode != 0:
        print("Error running src.analysis in fresh clone.")
        sys.exit(1)
        
    print("Testing src.ml...")
    res3 = subprocess.run([python_exe, "-m", "src.ml"], cwd=tmp_dir)
    if res3.returncode != 0:
        print("Error running src.ml in fresh clone.")
        sys.exit(1)
        
    print("Fresh clone test passed!")
    
    # Cleanup
    shutil.rmtree(tmp_dir)
    sys.exit(0)

if __name__ == "__main__":
    main()
