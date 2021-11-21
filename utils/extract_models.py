from pathlib import Path
import sys
import os
import shutil

def run(results_path):
    if(not Path('./pytorch_models').exists()):
            os.mkdir('./pytorch_models')
    out_dir = Path('./pytorch_models')
    dir = Path(results_path)
    for dataset_dir in dir.iterdir():
        if dataset_dir.is_dir():
            for method_dir in dataset_dir.iterdir():
                if method_dir.is_dir():
                    name = method_dir.name.lower()
                    source = method_dir.joinpath('generator.pt')
                    destination = out_dir.joinpath(name).with_suffix('.pt')
                    shutil.copyfile(str(source), str(destination))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        print('defaut location ../results used')
        run('../results')