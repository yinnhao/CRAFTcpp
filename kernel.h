#ifndef SUBTITLE_KERNEL_H_
#define SUBTITLE_KERNEL_H_
// #include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <cuda_fp16.h>
#include <unistd.h>
#include <stdexcept>
#include <string.h>
#include <fstream>
#include <unistd.h>
// #include <cstdint>
#include <fcntl.h>
#include "component.h"
// using namespace std;
void yuv444p2rgb_709(float* yuv, half* rgb, int h, int w);
void initbuf(char *filename, void *bin, long size);
void dumpbin(char *name, void *bin, int len);
void nv12_yConvertTo(float *dst, uint8_t *src, float scale, float low_value, int width, int height, int line_size);
void nv12_uvConvertTo(float *dst_u, float *dst_v, uint8_t *src, float scale, float low_value, int width, int height, int scale_width, int scale_height, int line_size);
void doscale(cudaStream_t stream, int batch, half *b, half *c, int input_height, int input_width, int output_height, int output_width);
void pad(half* in, half* out, int ori_h, int ori_w, int new_h, int new_w);
void threshold_map(half *score_map, float *text_score, float *link_score, uint8_t *text_score_comb, float text_threshold, float link_threshold, float low_text, int h, int w);
void half2float(half* in, float* out, int h, int w);
void rgb_norm(uint8_t *rgb_i, half* rgb, int h, int w);
void dilate(unsigned char * src, unsigned char *step1, int radio, int h, int w);
void get_box_from_label(unsigned char *d_image, unsigned int *d_labels, ConnectedComponent *components, ConnectedComponent *components_real, ConnectedComponent *components_real_h, int *count, int *count_h, unsigned int X, unsigned int Y);
#endif