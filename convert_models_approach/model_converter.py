import torch
from torch.autograd import Variable
import onnx
from onnx_tf.backend import prepare
from generators import GeneratorNet, GeneratorNetCifar10
from pathlib import Path
import sys

def convert_filepath(file_name, ending):
    if(file_name != ''):
        return file_name.split('.')[0] + ending

def torchToTf(file_path):
    if 'cifar' in file_path:
        trained_model = GeneratorNetCifar10()
        dummy_input = Variable(torch.randn(1, 100, 1, 1))
    else:
        trained_model = GeneratorNet()
        dummy_input = Variable(torch.randn(1, 100))

    trained_model.load_state_dict(torch.load(file_path))
    onnx_filepath = convert_filepath(file_path, '.onnx')
    torch.onnx.export(trained_model, dummy_input, onnx_filepath)
    
    model_onnx = onnx.load(onnx_filepath)
    tf_rep = prepare(model_onnx)
    tf_filepath = convert_filepath(file_path, '.pb')
    tf_rep.export_graph(tf_filepath)

def get_filepaths(dir_path = '.'):
    directory = Path(dir_path)
    filepaths = []
    for item in directory.iterdir():
        if item.is_file() and str(item).endswith('.pt'):
            filepaths.append(str(item))
    return filepaths

def convert_all_models(dir_path = '.'):
    filepaths = get_filepaths(dir_path)
    for file in filepaths:
        torchToTf(file)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        convert_all_models(sys.argv[1])
    else:
        print('models in current directory converted')
        convert_all_models()