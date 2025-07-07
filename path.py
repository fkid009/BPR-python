import os

class Path:
    MAIN_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    DATA_PATH = os.path.join(MAIN_PATH, "data")
