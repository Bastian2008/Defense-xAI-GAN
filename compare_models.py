from generators import GeneratorNet, GeneratorNetCifar10
import torch
import tensorflow as tf
from pathlib import Path
import sys

def test_torch(model_path, input):
    if 'cifar' in model_path:
        trained_model = GeneratorNetCifar10()
    else:
        trained_model = GeneratorNet()
    trained_model.load_state_dict(torch.laod(model_path))
    trained_model.eval()
    out = trained_model(input)
    return out

def test_tf(model_path, input):
    trained_model = tf.keras.models.load_model(model_path) 
    out = trained_model.predict(input)
    return out

def compare_outputs(torch_model_path, tf_model_path):
    input = torch.randn(1,100,1,1) if 'cifar' in torch_model_path else torch.randn(1,100)
    out_torch = test_torch(torch_model_path, input)
    out_tf = test_tf(tf_model_path, input)
    path = Path(tf_model_path)
    out_file  = open(f'{path.parent}/{path.name}_output.txt', 'w')
    out_file.write(f'Output Pytorch: {out_torch}\n')
    out_file.write(f'Output Tensorflow: {out_tf}\n')
    out_file.write(f'Input variable: {input}')
    out_file.close()
    print(out_torch == out_tf)

def check_paths(torch_path, tf_path):
    torch_model = Path(torch_path)
    tf_model = Path(tf_path)
    # check ending
    if torch_model.suffix != '.pt' or torch_model.suffix != '.pth':
        return False
    if tf_model.suffix != '.pb' or tf_model.suffix != '.h5':
        return False
    # check existance
    if not torch_model.exists():
        return False
    if not tf_model.exists():
        return False
    return True

if __name__ == '__main__':
    if len(sys.argv) > 3:
        if check_paths(sys.argv[1], sys.argv[2]):
            compare_outputs(sys.argv[1], sys.argv[2])
        else:
            print('You must pass as arguments:\nArgument 1: path to a pytorch model\nArgumennt 2: path to a tensorflow model')
    else:
        print('You must pass as arguments:\nArgument 1: path to a pytorch model\nArgumennt 2: path to a tensorflow model')
    