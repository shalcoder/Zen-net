
try:
    import cv2
    import tensorflow
    import requests
    print("Imports successful")
except ImportError as e:
    print(f"Import failed: {e}")
