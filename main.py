# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib
# pyrefly: ignore [missing-import]
import ultralytics

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Ultralytics:", ultralytics.__version__)

print("ANPR environment is working!")