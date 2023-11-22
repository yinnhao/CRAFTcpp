import numpy as np
import random
import cv2
def read_yuv(path, shape, dtype):
    """
    Defines read yuv from disk function

    Parameters
    ----------
    path: 
        file name
        string
    shape:
        the shape of yuv data
        list
        [c, h, w]
    dtype:
        the data type of yuv
        np.uint8 or np.uint16() 

    Returns
    -------
    data:

        Corresponding: yuv value read from file
        numpy array
        shape: [c, h, w]
        

    """
    size = 1
    for i in shape:
        size = size*i
    fn_lr = open(path, 'rb')
    data = np.fromfile(fn_lr, dtype=dtype, count=size).reshape(shape)
    fn_lr.close()
    return data


def label_vis(label_np):
    label_list = label_np.flatten().tolist()
    label_set = list(set(label_list))
    label_num = len(label_set)
    print(label_num)
    r, g, b = list(range(0,256)), list(range(0,256)), list(range(0,256))
    random.shuffle(r), random.shuffle(g), random.shuffle(b)
    # if label_num > 256:
    #     raise Exception("label num more than 255, can't visilize")
    h, w = label_np.shape[0], label_np.shape[1]
    vis_mask = np.zeros((h, w, 3), dtype=np.uint8)
    r_index, g_index, b_index = 0, 0, 0
    for i in range(label_num):
        vis_mask[:, :, 0][label_np == label_set[i]] = r[r_index]
        vis_mask[:, :, 1][label_np == label_set[i]] = g[g_index]
        vis_mask[:, :, 2][label_np == label_set[i]] = b[b_index]
        b_index += 1
        if b_index > 255:
            b_index = 0
            g_index += 1
        if g_index > 255:
            g_index = 0
            r_index += 1
        if r_index > 255:
            break
        # print(r_index, g_index, b_index)
    return vis_mask

# path = "pad_2.rgb"
# yuv = read_yuv(path, (3, 1088, 1920), np.float32)
# print(yuv.shape)
# rgb = np.transpose(yuv, (1, 2, 0))

# mean=(0.485, 0.456, 0.406)
# variance=(0.229, 0.224, 0.225)

# rgb *= np.array([variance[0] , variance[1], variance[2]], dtype=np.float32)
# rgb += np.array([mean[0] , mean[1], mean[2]], dtype=np.float32)

# rgb_i = np.clip(rgb * 255, 0, 255).astype(np.uint8)
# cv2.imwrite("pad_i_2080_2.png", rgb_i[:,:,::-1])
# print(rgb)


# path = "rgb.rgb"
# yuv = read_yuv(path, (3, 2160, 3840), np.float32)
# print(yuv.shape)
# rgb = np.transpose(yuv, (1, 2, 0))
# print(rgb)

# mean=(0.485, 0.456, 0.406)
# variance=(0.229, 0.224, 0.225)

# rgb *= np.array([variance[0] , variance[1], variance[2]], dtype=np.float32)
# rgb += np.array([mean[0] , mean[1], mean[2]], dtype=np.float32)

# rgb_i = np.clip(rgb * 255, 0, 255).astype(np.uint8)
# cv2.imwrite("rgb.png", rgb_i[:,:,::-1])

# path = "y.yuv"
# yuv = read_yuv(path, (2160, 3840), np.uint8)
# cv2.imwrite("y.png", yuv)


path = "yuv444p.yuv"
yuv = read_yuv(path, (3, 2160, 3840), np.float32)
yuv[0,:,:] = yuv[0,:,:]*219+16
yuv[1,:,:] = yuv[1,:,:]*224+16
yuv[2,:,:] = yuv[2,:,:]*224+16
yuv = yuv.astype(np.uint8)
yuv.tofile("yuv444p_2.yuv")
# cv2.imwrite("yuv.png", yuv)

# print(rgb)

# path = "/data/QCVLib/QTD/cpp/score_int.bin"
# score_map = read_yuv(path, (544, 960), np.int8)
# score_map_i = (score_map*200).astype(np.uint8)
# cv2.imwrite("./score_int_2.png", score_map_i)


# path = "./score_2_2080.bin"
# score_map = read_yuv(path, (544, 960, 2), np.float32)
# score_map_i = (np.clip(score_map[:,:,0]*255, 0, 255)).astype(np.uint8)
# cv2.imwrite("./score_2_2080_2.png", score_map_i)

# path = "/data/adjust_subtitle_lum/subtitle_label_map_2.yuv"
# label = read_yuv(path, (2160, 3840), np.uint32)
# vis = label_vis(label)
# # vis[437, 431:529, :] = 0
# # vis[455, 431:529, :] = 0
# # vis[437:455, 431, :] = 0
# # vis[437:455, 529, :] = 0
# cv2.imwrite("./subtitle_label_map_2.png", vis)


# path = "/data/QCVLib/QTD/cpp/cur_mask.bin"
# score_map = read_yuv(path, (1080, 1920), np.uint8)
# score_map_i = (score_map*255).astype(np.uint8)
# cv2.imwrite("/data/adjust_subtitle_lum/cur_mask.png", score_map_i)

# path = "/data/QCVLib/QTD/cpp/y_full_range_d.bin"
# score_map = read_yuv(path, (1080, 1920), np.uint8)
# score_map_i = (score_map).astype(np.uint8)
# cv2.imwrite("/data/adjust_subtitle_lum/y_full_range_d.png", score_map_i)