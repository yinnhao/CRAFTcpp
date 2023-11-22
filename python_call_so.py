'''
@File    :   python_call_so.py
@Time    :   2023/08/31 15:33:26
@Author  :   zhuyinghao
@Desc    :   提供从python调用detector.so的接口,可以读入图像，输出带文字框的图像结果
'''
import ctypes
import numpy as np
import cv2
import os
import cupy as cp
import logging
lib = ctypes.CDLL("/data/qtd_cpp/cpp/libdetector.so")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='print_box_center_x.log', level=logging.DEBUG, format=LOG_FORMAT)

print(lib.infer_init)
print(lib.infer_pipe)

def read_yuv(path, shape, dtype):
    size = 1
    for i in shape:
        size = size*i
    fn_lr = open(path, 'rb')
    data = np.fromfile(fn_lr, dtype=dtype, count=size).reshape(shape)
    fn_lr.close()
    return data

def is_subtitle_in_center(x_min, x_max, center_x):
    center_point = x_min + (x_max - x_min) / 2
    print("center point:", center_point)
    logging.info("center point diff:{}".format(str(abs(center_x-center_point))))

onnx_path = "/data/QCVLib/QTD/pth_model/epoch_91_fp16.onnxdynamic_shape.cache";
in_format = "yuv"
if in_format == "yuv":
    # path = "./dakao_fuqin.yuv"
    path = "/data/adjust_subtitle_lum/baobao_nv12.yuv"
    print(path)
    # path = "./4k-B_fengqiluoyang_17min_27min_toufa_nv12.yuv"
    h, w = 2160, 3840
    # engine_path = "../pth_model/epoch_91_fp16.onnx_0821.trt";
    # onnx_path = "../pth_model/epoch_91_fp16.onnx";

    yuv = read_yuv(path, (int(h/2*3), w), np.uint8)
    if not yuv.flags['C_CONTIGUOUS']:
        yuv = np.ascontiguousarray(yuv, dtype=yuv.dtype)

    yuv_cp = cp.asarray(yuv)
    # lib.infer_test(h, w)
    ratio = 0.5
    lib.infer_init(h, w, ctypes.c_char_p(onnx_path.encode('utf-8')), ctypes.c_float(ratio))
    # lib.infer_pipe(yuv.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    size = ctypes.c_int()
    data_ptr = ctypes.POINTER(ctypes.c_int)()

    format = 1
    lib.python_api(ctypes.c_void_p(yuv_cp.data.ptr), ctypes.byref(size), ctypes.byref(data_ptr), format)

    # Convert the returned C array to a NumPy array
    returned_data = np.ctypeslib.as_array(data_ptr, shape=(size.value,))
    print(type(returned_data))
    print(returned_data.shape)
    print("Returned data:", returned_data)
else:
    
    def read_img_and_plot_rect(path, save_folder): 
        print(path)
        logging.info(path)
        img = cv2.imread(path)
        file_name = os.path.basename(path)
        img = img[:,:,::-1]
        img = np.transpose(img, (2, 0, 1))
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img, dtype=img.dtype)

        format = 0
        h, w = img.shape[1], img.shape[2]
        center_x = int(w/2)
        print("image real center width:", center_x)
        logging.info("image real center width:{}".format(str(center_x)))
        ratio = 0.5
        # if h > 1280:
        #     ratio = 0.25
        lib.infer_init(h, w, ctypes.c_char_p(onnx_path.encode('utf-8')), ctypes.c_float(ratio))

        img_cp = cp.asarray(img)

        size = ctypes.c_int()
        data_ptr = ctypes.POINTER(ctypes.c_int)()

        lib.python_api(ctypes.c_void_p(img_cp.data.ptr), ctypes.byref(size), ctypes.byref(data_ptr), format)
        # lib.python_api(img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), ctypes.byref(size), ctypes.byref(data_ptr), format)

        # Convert the returned C array to a NumPy array
        returned_data = np.ctypeslib.as_array(data_ptr, shape=(size.value,))

        n = returned_data.shape[0]

        print(type(returned_data))
        print(returned_data.shape[0])
        print("Returned data:", returned_data)

        

        img = cv2.imread(path)
        for i in range(0, n, 4):
            is_subtitle_in_center(int(returned_data[i+0]), int(returned_data[i+1]), center_x)
            tl_point = [int(returned_data[i+0]), int(returned_data[i+2])]
            br_point = [int(returned_data[i+1]), int(returned_data[i+3])]
            cv2.rectangle(np.ascontiguousarray(img), tl_point, br_point, (0,0,255), 2)
        cv2.imwrite("{}/{}_det.jpg".format(save_folder, file_name.split(".")[0]), img)
        lib.destroyObj()

    path = "/data/QCVLib/QTD/test_img/zhuqinghao_p18_bug_2.png"
    # path = "../test_img"
    # path = "/mnt/ec-data2/ivs/1080p/zyh/text_detect/text_dataset/rengong_badcase_src"
    # path = "/mnt/ec-data2/ivs/1080p/zyh/zimu_detect/yizi"
    save_folder = "./res3"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # 是不是图片
    if os.path.splitext(path)[-1][1:] == "png":
        read_img_and_plot_rect(path, save_folder)
    else:
        file_list = os.listdir(path)
        for file in file_list:
            read_img_and_plot_rect(os.path.join(path, file), save_folder)