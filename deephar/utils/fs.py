import os

def mkdir(path):
    if os.path.isdir(path) is False:
        os.mkdir(path)

