# main.py

import subprocess
import sys

def start_interview():
    print("Starting AI Interview Session...\n")
    subprocess.run([sys.executable, "-m", "frontend.minimal"])

if __name__ == "__main__":
    start_interview()