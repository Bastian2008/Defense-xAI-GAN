import shutil
from pathlib import Path
import os

def organize_directory(dir_path = '.'):
    items_list = list(Path(dir_path).glob('?*.*'))
    create_directories(items_list, dir_path)
    for file in items_list:
        print(file)
        parent = file.parent
        name = file.name
        ending =  file.suffix[1:]
        shutil.move(str(file), f'{parent}/{ending}/{name}')

def create_directories(file_list, dirPath):
    dir_names = set()
    for file in file_list:
        dir_names.add(file.suffix[1:])
    for dir in dir_names:
        if(not Path(f'{dirPath}/{dir}').exists()):
            os.mkdir(f'{dirPath}/{dir}')