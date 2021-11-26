from generators import GeneratorNet, GeneratorNetCifar10
import numpy as np
import torch
import tensorflow as tf
from pathlib import Path
import sys
# np.set_printoptions(threshold = 100000)

input = torch.tensor([[-0.7437, -1.2760, -0.1883, -0.1868,  0.1205, -0.4901, -0.8610,  0.8493,
         -0.4686,  1.2384, -0.4359, -1.9974,  0.2344,  0.3692,  0.0768,  0.7325,
          0.8549, -1.0880,  0.1584,  0.6923,  0.5956,  2.1244, -0.2668, -1.9012,
         -1.2560,  0.0883, -0.8099, -1.1211, -1.9375,  0.1645,  0.7323,  0.4977,
          0.2530,  1.4484,  0.5619,  0.6578, -0.3100, -0.4125,  0.0383, -0.5223,
          1.0584, -2.2717, -0.4142,  0.1609, -1.7738,  0.4033,  0.8331, -1.3342,
         -0.8637,  1.1586, -0.5320, -0.3415, -0.2445,  1.4211, -0.6328, -2.0225,
         -0.6776,  3.5800,  0.3418,  2.0807, -1.2033,  0.8652,  1.0323, -0.8722,
          0.3577,  1.2549, -1.0873,  0.1367, -1.0455, -1.4343,  0.1087,  0.7151,
         -0.8155, -1.6259, -0.4599, -1.1088, -0.1051, -1.1991, -0.2992, -0.7636,
         -0.9617,  0.2370, -0.1049,  1.1404,  0.3503, -1.7872,  1.0915, -1.7883,
          0.3270, -1.8282, -0.6075, -0.0076, -0.5324,  1.4404, -1.2606, -0.7243,
          0.2621,  1.3987, -0.3303,  0.0961]])

def test_torch(model_path, input):
    with torch.no_grad():
        if 'cifar' in model_path:
            trained_model = GeneratorNetCifar10()
        else:
            trained_model = GeneratorNet()

        state_dict = torch.load(model_path, map_location='cpu')
        trained_model.load_state_dict(state_dict)
        out = trained_model(input)
        print(out) #Just for testing
        return out

def test_tf(model_path, input):
    print('testing tensorflow model')
    loaded_model = tf.saved_model.load(model_path) 
    infer = loaded_model.signatures['serving_default']
    out = infer(input)
    print(out)  #Just for testing
    return out

def compare_outputs(torch_model_path, tf_model_path):
    torch_input = input #torch.randn(1,100,1,1) if 'cifar' in torch_model_path else torch.randn(1,100)
    tf_input = tf.constant(torch_input)
    out_torch = test_torch(torch_model_path, torch_input)
    out_tf = test_tf(tf_model_path, tf_input)
    path = Path(torch_model_path)
    model_name = path.name[:path.name.find('.')]
    out_file = open(f'{path.parent}/{model_name}_output.txt', 'w')
    out_file.write(f'Output Pytorch: {out_torch}')
    out_file.write(f'Output Tensorflow: {out_tf}')
    out_file.write(f'Input variable: {torch_input}')
    bool_arr = np.isclose(out_torch.numpy(), out_tf['26'].numpy(), rtol=1e-04, atol=1e-08, equal_nan=False)
    out_file.write(f'Comparison array: {bool_arr}')
    out_file.close()
    print(np.all(bool_arr))

def check_paths(torch_path, tf_path):
    torch_model = Path(torch_path)
    tf_model = Path(tf_path)
    # check ending
    if torch_model.suffix != '.pt' and torch_model.suffix != '.pth':
        print('not pytorch suffix')
        return False
    if tf_model.suffix != '.pb' and tf_model.suffix != '.h5':
        print('not tensorflow suffix')
        return False
    # check existance
    if not torch_model.exists():
        print('Pytorch model file does not exist')
        return False
    if not tf_model.exists():
        print('Tensorflow model file does not exist')
        return False
    return True

if __name__ == '__main__':
    if len(sys.argv) > 2:
        if check_paths(sys.argv[1], sys.argv[2]):
            compare_outputs(sys.argv[1], sys.argv[2])
        else:
            print('You must pass as arguments:\nArgument 1: path to a pytorch model\nArgumennt 2: path to a tensorflow model')
    else:
        print('less arguments than expected. You must pass as arguments:\nArgument 1: path to a pytorch model\nArgumennt 2: path to a tensorflow model')
    