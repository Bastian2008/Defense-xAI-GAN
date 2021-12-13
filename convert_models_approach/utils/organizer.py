import shutil
from pathlib import Path
import os
import sys

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

def extract_cifar_models(dir_path = '.'):
    dir = Path(dir_path).resolve()
    prev_dir = dir.parent
    for item in dir.iterdir():
        old_path = str(item)
        if item.is_file() and 'cifar' in old_path:
            old_path = str(item)
            new_path = str(prev_dir.joinpath(item.name))
            shutil.move(old_path, new_path)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        extract_cifar_models(sys.argv[1])
        # organize_directory(sys.argv[1])
    else:
        print('current directory organized')
        extract_cifar_models()
        # organize_directory()