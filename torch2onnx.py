'''
@File    :   torch2onnx.py
@Time    :   2023/03/24 14:35:03
@Author  :   zhuyinghao
@Desc    :   将craft模型转成onnx模型
'''
import os
import onnx
import torch
import numpy as np
import argparse
from craft import CRAFT
from collections import OrderedDict

def to_numpy(tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        return tensor.cpu().numpy()

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='torch2onnx')
    parser.add_argument('--usefp16', action="store_true", default=False)
    parser.add_argument('--torch_path', type=str)
    args = parser.parse_args()

    # model_path = "weights"
    usefp16 = args.usefp16
    if usefp16:
        print("use fp16 precision")
    else:
        print("use fp32 precision")
    trained_model_path = args.torch_path
    suffix = "_fp16" if usefp16 else "_fp32"
    
    folder_path = os.path.dirname(trained_model_path)
    model_name = os.path.basename(trained_model_path)
    
    save_onnx_path = os.path.join(folder_path, model_name.split(".")[0] + suffix + ".onnx")
    
    # 定义网络
    net = CRAFT()
    # load weights
    print('Loading weights from checkpoint (' + trained_model_path + ')')
    net.load_state_dict(copyStateDict(torch.load(trained_model_path, map_location='cpu')))
    net = net.cuda()
    net.eval()
    # 定义输入
    x = torch.rand((1, 3, 1088, 1920), dtype=torch.float32)
    x = x.cuda()
    
    print('save onnx to (' + save_onnx_path + ')')
    with torch.no_grad():
        if usefp16:
            net = net.half()
            x = x.half()
        x_onnx = to_numpy(x)
        y, fea = net(x)
        print(y)
        print("save onnx to {}".format(save_onnx_path))
        torch.onnx.export(net, x, save_onnx_path, export_params=True, opset_version=11, do_constant_folding=True, input_names=['input1'],output_names=['output'],dynamic_axes={'input1': {2: "height", 3: "width"}},verbose=True)
    


    # 验证onnx
    import onnxruntime

    # verify exported onnx model
    detector_onnx = onnx.load(save_onnx_path)
    onnx.checker.check_model(detector_onnx)
    print(f"Model Inputs:\n {detector_onnx.graph.input}\n{'*'*80}")
    print(f"Model Outputs:\n {detector_onnx.graph.output}\n{'*'*80}")
    ort_session = onnxruntime.InferenceSession(save_onnx_path)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: x_onnx}
    y_onnx_out, feature_onnx_out = ort_session.run(None, ort_inputs)
    # print(y_onnx_out.dtype)
    print(f"torch outputs: y_torch_out.shape={y.shape} feature_torch_out.shape={fea.shape}")
    print(f"onnx outputs: y_onnx_out.shape={y_onnx_out.shape} feature_onnx_out.shape={feature_onnx_out.shape}")
    
        
 