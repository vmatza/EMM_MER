import os

def make_output_dirs():
    """
    Create project output directories if they do not exist.
    """
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)