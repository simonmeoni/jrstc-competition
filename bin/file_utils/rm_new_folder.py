import os
import shutil


def rm_and_new_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=False)
    os.makedirs(path)
