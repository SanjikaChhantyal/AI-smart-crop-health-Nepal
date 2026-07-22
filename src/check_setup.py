"""Check whether the required project libraries are installed correctly."""

import sys

import cv2
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import streamlit
from PIL import Image


def main() -> None:
    """Display the installed Python and library versions."""

    print("Project environment check")
    print("-------------------------")
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"Streamlit: {streamlit.__version__}")
    print(f"Pillow: {Image.__version__}")
    print("\nAll required libraries imported successfully!")


if __name__ == "__main__":
    main()