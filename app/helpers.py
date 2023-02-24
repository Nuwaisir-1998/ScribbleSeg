import os

def make_directory_if_not_exist(path):
        if not os.path.exists(path):
            os.makedirs(path)