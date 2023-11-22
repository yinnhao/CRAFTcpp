
#include "kernel.h"
#include "component.h"
#define BLOCKDIMX 32
#define BLOCKDIMY 16

__forceinline__ static float area_pixel_compute_scale(
    int input_size,
    int output_size) {
if (output_size > 1) {
    return  (float)input_size / output_size;
} else {
    return 0;
}
}

__device__ __forceinline__ float area_pixel_compute_source_index(
    float scale,
    int dst_index) {

float src_idx = scale * ((float)dst_index + 0.5) - 0.5;
if (src_idx < 0)
    src_idx = 0;

return src_idx;
} 

__device__ float wr = 0.2126;
__device__ float wg = 0.7152;
__device__ float wb = 0.0722;
__device__ float min_luma= 0; 
__device__ float max_luma= 1.0; //235
__device__ float min_chroma = 0; //16
__device__ float max_chroma = 1.0; //240
__device__ float y_offset = 0;
__device__ float u_offset = -0.5; 
__device__ float v_offset = -0.5; //-512;
__device__ float max_value= 1.0; //1023.0;
__device__ float rgbMean[3] = {0.485, 0.456, 0.406};
__device__ float rgbVar[3] = {0.229, 0.224, 0.225};


__global__ void rgb_norm_kernel(uint8_t *rgb_i, half* rgb, int h, int w, int input_h_stride, int input_c_stride){
    const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
    if(w_i<w && h_i<h){
        float r = (float)rgb_i[0*input_c_stride+h_i*input_h_stride+w_i] / 255.0;
        float g = (float)rgb_i[1*input_c_stride+h_i*input_h_stride+w_i] / 255.0;
        float b = (float)rgb_i[2*input_c_stride+h_i*input_h_stride+w_i] / 255.0;
        r = (r - rgbMean[0]) / rgbVar[0];
        g = (g - rgbMean[1]) / rgbVar[1];
        b = (b - rgbMean[2]) / rgbVar[2];
        rgb[0*input_c_stride+h_i*input_h_stride+w_i] = (half)r;
        rgb[1*input_c_stride+h_i*input_h_stride+w_i] = (half)g;
        rgb[2*input_c_stride+h_i*input_h_stride+w_i] = (half)b;
    }
}
void rgb_norm(uint8_t *rgb_i, half* rgb, int h, int w){
    int GRIDDIMX = int((w + BLOCKDIMX - 1) / BLOCKDIMX);
    int GRIDDIMY = int((h + BLOCKDIMY - 1) / BLOCKDIMY);
    int GRIDDIMZ = 1;
    dim3 grid_size(GRIDDIMX, GRIDDIMY, 1);
    dim3 block_size(BLOCKDIMX, BLOCKDIMY, 1);
    rgb_norm_kernel<<<grid_size, block_size>>>(rgb_i, rgb, h, w, w, h*w);
}


__global__ void yuv444p2rgb_709_kernel(float* yuv, half* rgb, int h, int w, int input_h_stride, int input_c_stride){
    const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
    if(w_i<w && h_i<h){
        // if(w_i==0 && h_i==0){
        //     printf("%f\n", rgbMean[0]);
        // }
        float factor_0_0 = max_value/(max_luma - min_luma);
        float factor_0_1 = 0;
        float factor_0_2 = max_value/(max_chroma - min_chroma)*(2-2*wr);
        float factor_1_0 = max_value/(max_luma - min_luma);
        float factor_1_1 = -max_value/(max_chroma - min_chroma)*(2-2*wb) *wb/wg;
        float factor_1_2 = -max_value/(max_chroma - min_chroma)*(2-2*wr)*wr/wg;
        float factor_2_0 = max_value/(max_luma - min_luma);
        float factor_2_1 = max_value/(max_chroma - min_chroma)*(2-2*wb);
        float factor_2_2 = 0;

        float y = yuv[0*input_c_stride+h_i*input_h_stride+w_i];
        float u = yuv[1*input_c_stride+h_i*input_h_stride+w_i];
        float v = yuv[2*input_c_stride+h_i*input_h_stride+w_i];

        float r, g, b;
        r = factor_0_0*(y+y_offset) + factor_0_1*(u+u_offset) + factor_0_2*(v+v_offset);
        g = factor_1_0*(y+y_offset) + factor_1_1*(u+u_offset) + factor_1_2*(v+v_offset);
        b = factor_2_0*(y+y_offset) + factor_2_1*(u+u_offset) + factor_2_2*(v+v_offset);

        if(r>1) r=1; if(r<0) r=0;
        if(g>1) g=1; if(g<0) g=0;
        if(b>1) b=1; if(b<0) b=0;

        r = (r - rgbMean[0]) / rgbVar[0];
        g = (g - rgbMean[1]) / rgbVar[1];
        b = (b - rgbMean[2]) / rgbVar[2];

        rgb[0*input_c_stride+h_i*input_h_stride+w_i] = (half)r;
        rgb[1*input_c_stride+h_i*input_h_stride+w_i] = (half)g;
        rgb[2*input_c_stride+h_i*input_h_stride+w_i] = (half)b;

    }

}

void yuv444p2rgb_709(float* yuv, half* rgb, int h, int w){
    int GRIDDIMX = int((w + BLOCKDIMX - 1) / BLOCKDIMX);
    int GRIDDIMY = int((h + BLOCKDIMY - 1) / BLOCKDIMY);
    int GRIDDIMZ = 1;
    dim3 grid_size(GRIDDIMX, GRIDDIMY, 1);
    dim3 block_size(BLOCKDIMX, BLOCKDIMY, 1);
    yuv444p2rgb_709_kernel<<<grid_size, block_size>>>(yuv, rgb, h, w, w, h*w);
}

void initbuf(char *filename, void *bin, long size) {
	int fd = open(filename, O_RDONLY);
	int n = read(fd, bin, size);
	close(fd);
	// cudaMemcpy(d, bin, n, cudaMemcpyHostToDevice);
}

void dumpbin(char *name, void *bin, int len) {
	// cudaMemcpy(bin, addr, len, cudaMemcpyDeviceToHost);
	int fd = open(name, O_CREAT | O_WRONLY | O_TRUNC, 0666);
	write(fd, bin, len);
	close(fd);
}

__global__ void nv12_yConvertTo_kernel(float *dst, uint8_t *src, float scale, float low_value, int width, int height, int line_size)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < width && i < height)
    {
       float value = (src[i * line_size + j] - low_value) * scale;
        if (value < 0)
            value = 0;
        else if (value > 1)
            value = 1;
        dst[i * width + j] = value;
    }
}

void nv12_yConvertTo(float *dst, uint8_t *src, float scale, float low_value, int width, int height, int line_size)
{
    int GRIDDIMX = int((width + BLOCKDIMX - 1) / BLOCKDIMX);
    int GRIDDIMY = int((height + BLOCKDIMY - 1) / BLOCKDIMY);
    dim3 grid_size(GRIDDIMX, GRIDDIMY, 1);
    dim3 block_size(BLOCKDIMX, BLOCKDIMY, 1);
    nv12_yConvertTo_kernel<<<grid_size, block_size>>>(dst, src, scale, low_value, width, height, line_size);     
}

__global__ void nv12_uvConvertTo_kernel(float *dst_u, float *dst_v, uint8_t *src, float scale, float low_value, int width, int height, int scale_width, int scale_height, int line_size)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < scale_width && i < scale_height)
    {
        int src_pos = i * line_size + j * 2;
        int dst_pos = (2 * i) * width + 2 * j;

        float value = (src[src_pos] - low_value) * scale;
        if (value < 0)
            value = 0;
        else if (value > 1)
            value = 1;

        dst_u[dst_pos] = value;
        dst_u[dst_pos + 1] = value;
        dst_u[dst_pos + width] = value;
        dst_u[dst_pos + width + 1] = value;

        value = (src[src_pos + 1] - low_value) * scale;
        if (value < 0)
            value = 0;
        else if (value > 1)
            value = 1;
        dst_v[dst_pos] = value;
        dst_v[dst_pos + 1] = value;
        dst_v[dst_pos + width] = value;
        dst_v[dst_pos + width + 1] = value;
    }
}

void nv12_uvConvertTo(float *dst_u, float *dst_v, uint8_t *src, float scale, float low_value, int width, int height, int scale_width, int scale_height, int line_size)
{
    int GRIDDIMX = int((scale_width + BLOCKDIMX - 1) / BLOCKDIMX);
    int GRIDDIMY = int((scale_height + BLOCKDIMY - 1) / BLOCKDIMY);
    dim3 grid_size(GRIDDIMX, GRIDDIMY, 1);
    dim3 block_size(BLOCKDIMX, BLOCKDIMY, 1);
    nv12_uvConvertTo_kernel<<<grid_size, block_size>>>(dst_u, dst_v, src, scale, low_value, width, height, scale_width, scale_height, line_size);     
}




__global__ void upsample_bilinear2d_out_frame_half(
    const int n,
    const float rheight,
    const float rwidth,
    const int batchsize,
    const int channels,
    const int height1,
    const int width1,
    const int height2,
    const int width2,
    half *idata,
    half *odata) {
int index = threadIdx.x + blockIdx.x * blockDim.x;

if (index < n) {
    const int w2 = index % width2; // 0:width2-1
    const int h2 = index / width2; // 0:height2-1
    // special case: just copy
    if (height1 == height2 && width1 == width2) {
        const int h1 = h2;
        const int w1 = w2;
        for (int n = 0; n < batchsize; n++) {
            for (int c = 0; c < channels; ++c) {
                int src = n * channels* height1 * width1 + c * height1 * width1 + h1 * width1 + w1;
                int dst = n * channels* height2 * width2 + c * height2 * width2 + h2 * width2 + w2;
                const half val = idata[src];
                odata[dst] = val;
            }
        }
        return;
    }
    //
    const float h1r = area_pixel_compute_source_index(rheight, h2);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const float h1lambda = h1r - (float)h1;
    const float h0lambda = static_cast<float>(1) - h1lambda;
    //
    const float w1r = area_pixel_compute_source_index(rwidth, w2);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const float w1lambda = w1r - (float)w1;
    const float w0lambda = static_cast<float>(1) - w1lambda;
    //
    for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
            int src = n * channels* height1 * width1 + c * height1 * width1 + w1 + h1*width1;
            int src0 = src;
            int src1 = src + w1p;
            int src2 = src + h1p * width1;
            int src3 = src + h1p* width1+ w1p;
            int dst = n * channels* height2 * width2 + c * height2 * width2 + h2 * width2 + w2;
            const float val = h0lambda *
                (w0lambda * (float)idata[src0] +
                 w1lambda * (float)idata[src1]) +
                h1lambda *
                (w0lambda * (float)idata[src2] +
                 w1lambda * (float)idata[src3]);

            odata[dst] = static_cast<half>(val);
        }
    }
}
}

void doscale(cudaStream_t stream, int batch, half *b, half *c, int input_height, int input_width, int output_height, int output_width) {
	cudaError_t err;

	int channels;
	channels = 3;

	const int num_kernels = output_height * output_width;
	int num_threads;
	num_threads = 600;

	const float rheight = area_pixel_compute_scale(
			input_height, output_height);
	const float rwidth = area_pixel_compute_scale(
			input_width, output_width);

	int blocks = (num_kernels + num_threads - 1) / num_threads;
	upsample_bilinear2d_out_frame_half<<< blocks, num_threads, 0, stream>>>(num_kernels, rheight, rwidth, batch, channels, input_height, input_width, output_height, output_width, b, c);
}

__global__ void half2float_kernel(half* in, float* out, int h, int w, int input_h_stride, int input_c_stride){
    const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
    if(w_i<w && h_i<h){
        out[0*input_c_stride+h_i*input_h_stride+w_i] = __half2float(in[0*input_c_stride+h_i*input_h_stride+w_i]);
        out[1*input_c_stride+h_i*input_h_stride+w_i] = __half2float(in[1*input_c_stride+h_i*input_h_stride+w_i]);
        // out[2*input_c_stride+h_i*input_h_stride+w_i] = __half2float(in[2*input_c_stride+h_i*input_h_stride+w_i]);
    }
}
void half2float(half* in, float* out, int h, int w){
    int GRIDDIMX = int((w + BLOCKDIMX - 1) / BLOCKDIMX);
    int GRIDDIMY = int((h + BLOCKDIMY - 1) / BLOCKDIMY);
    // int GRIDDIMZ = 1;
    dim3 grid_size(GRIDDIMX, GRIDDIMY, 1);
    dim3 block_size(BLOCKDIMX, BLOCKDIMY, 1);
    half2float_kernel<<<grid_size, block_size>>>(in, out, h, w, w, h*w);
}

__global__ void pad_kernel(half* in, half* out, int ori_h, int ori_w, int new_h, int new_w){
    const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
    if(w_i<ori_w && h_i<ori_h){
        out[0*new_w*new_h+h_i*new_w+w_i] = in[0*ori_w*ori_h+h_i*ori_w+w_i];
        out[1*new_w*new_h+h_i*new_w+w_i] = in[1*ori_w*ori_h+h_i*ori_w+w_i];
        out[2*new_w*new_h+h_i*new_w+w_i] = in[2*ori_w*ori_h+h_i*ori_w+w_i];
    }
    else if(w_i<new_w && h_i<new_h){
        out[0*new_w*new_h+h_i*new_w+w_i] = 0;
        out[1*new_w*new_h+h_i*new_w+w_i] = 0;
        out[2*new_w*new_h+h_i*new_w+w_i] = 0;   
    }
}
void pad(half* in, half* out, int ori_h, int ori_w, int new_h, int new_w){
    int GRIDDIMX = int((new_w + BLOCKDIMX - 1) / BLOCKDIMX);
    int GRIDDIMY = int((new_h + BLOCKDIMY - 1) / BLOCKDIMY);
    // int GRIDDIMZ = 1;
    dim3 grid_size(GRIDDIMX, GRIDDIMY, 1);
    dim3 block_size(BLOCKDIMX, BLOCKDIMY, 1);
    pad_kernel<<<grid_size, block_size>>>(in, out, ori_h, ori_w, new_h, new_w);
}

static int padh, padw;
static void getpad(int w, int h) {
	int _w, _h;
	padw = padh = 0;
	if (w != ((w>>5)<<5))
		padw = (((w>>5)+1)<<5) - w;
	if (h != ((h>>5)<<5))
		padh = (((h>>5)+1)<<5) - h;
}

__global__ void threshold_map_kernel(half *score_map, float *text_score, float *link_score, uint8_t *text_score_comb, float text_threshold, float link_threshold, float low_text, int h, int w){
    const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_i = blockIdx.y * blockDim.y + threadIdx.y;

    if(w_i<w && h_i<h){
        float text_score_value = __half2float(score_map[h_i*w*2+2*w_i+0]);
        float link_score_value = __half2float(score_map[h_i*w*2+2*w_i+1]);
        // if(h_i==446 && w_i==441){
        //     printf("text_score_value:%f\n", text_score_value);
        //     printf("link_score_value:%f\n", link_score_value);
        // }
        float text_thres = 0;
        float link_thres = 0;
        uint8_t text_and_link_thres = 0;
        if(text_score_value>=low_text)
            text_thres = 1;
        if(link_score_value>=link_threshold)
            link_thres = 1;
        if(text_thres+link_thres>=1)
            text_and_link_thres = 1;
        text_score[h_i*w + w_i] = text_thres;
        link_score[h_i*w + w_i] = link_thres;
        text_score_comb[h_i*w + w_i] = text_and_link_thres;
        // if(h_i==446 && w_i==441){
        //     printf("text_score:%f\n", text_score[h_i*w + w_i]);
        //     printf("link_score:%f\n", link_score[h_i*w + w_i]);
        //     printf("text_score_comb:%d\n", text_score_comb[h_i*w + w_i]);
        // }

    }
}
void threshold_map(half *score_map, float *text_score, float *link_score, uint8_t *text_score_comb, float text_threshold, float link_threshold, float low_text, int h, int w){
    int GRIDDIMX = int((w + BLOCKDIMX - 1) / BLOCKDIMX);
    int GRIDDIMY = int((h + BLOCKDIMY - 1) / BLOCKDIMY);
    // int GRIDDIMZ = 1;
    dim3 grid_size(GRIDDIMX, GRIDDIMY, 1);
    dim3 block_size(BLOCKDIMX, BLOCKDIMY, 1);
    threshold_map_kernel<<<grid_size, block_size>>>(score_map, text_score, link_score, text_score_comb, text_threshold, link_threshold, low_text, h, w);
}

__global__ void DilateStep1(unsigned char * src, unsigned char * dst, int radio, int h, int w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= h || x >= w) {
        return;
    }
    unsigned int start_j = max(x - radio, 0);
    unsigned int end_j = min(w - 1, x + radio);
    unsigned char value = 0;
    for (int j = start_j; j <= end_j; j++) {
        value = max(value, src[y * w + j]);
    }
    dst[y * w + x] = value;
}

__global__ void DilateStep2(unsigned char * src, unsigned char * dst, int radio, int h, int w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= h || x >= w) {
        return;
    }
	// printf("fuck %d\n", x);
    unsigned int start_i = max(y - radio, 0);
    unsigned int end_i = min(h - 1, y + radio);
    unsigned char value = 0;
    for (int i = start_i; i <= end_i; i++) {
        value = max(value, src[i * w + x]);
		
    }
    dst[y * w + x] = value;
}

void dilate(unsigned char * src, unsigned char *step1, int radio, int h, int w){

    int GRIDDIMX = int((w + BLOCKDIMX - 1) / BLOCKDIMX);
    int GRIDDIMY = int((h + BLOCKDIMY - 1) / BLOCKDIMY);
    dim3 grid_size(GRIDDIMX, GRIDDIMY, 1);
    dim3 block_size(BLOCKDIMX, BLOCKDIMY, 1);
    DilateStep1<<<grid_size,block_size>>>(src, step1, radio, h, w);
    cudaDeviceSynchronize();
    DilateStep2<<<grid_size,block_size>>>(step1, src, radio, h, w);
    
}
// 统计每个连通域的面积和上下定点坐标
__global__ void get_label_area(unsigned int *d_labels, unsigned char *d_image, ConnectedComponent *components, int cY, int cX){
	int h_i = (blockIdx.y * blockDim.y) + threadIdx.y;
	int w_i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(h_i<cY && w_i<cX){
		if(d_image[h_i*cX+w_i]==1){
			// atomicAdd(label_area+d_labels[h_i*cX+w_i], 1);
			atomicAdd(&components[d_labels[h_i*cX+w_i]].area, 1);
            // 因为startX和startY要求最小值，而初始化为0，所以这里用cX和cY减去原来的值，求最大值
			atomicMax(&components[d_labels[h_i*cX+w_i]].startX, cX-w_i);
			atomicMax(&components[d_labels[h_i*cX+w_i]].startY, cY-h_i);
			atomicMax(&components[d_labels[h_i*cX+w_i]].endX, w_i);
			atomicMax(&components[d_labels[h_i*cX+w_i]].endY, h_i);
		}
	}

}
// 处理components中的label，将面积大于10的label传输到components_real中的前count个元素中
__global__ void process_label(ConnectedComponent *components, ConnectedComponent *components_real, int *count, int cY, int cX){
	int h_i = (blockIdx.y * blockDim.y) + threadIdx.y;
	int w_i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(h_i<cY && w_i<cX){
		if(components[h_i*cX+w_i].area>=10){
			// atomicAdd(label_area+d_labels[h_i*cX+w_i], 1);
			int sx = cX - components[h_i*cX+w_i].startX;
			int sy = cY - components[h_i*cX+w_i].startY;
			int ex = components[h_i*cX+w_i].endX;
			int ey = components[h_i*cX+w_i].endY;
			int area = components[h_i*cX+w_i].area;
            // 将左上角的坐标转换回来
			components[h_i*cX+w_i].startX = cX - components[h_i*cX+w_i].startX;
			components[h_i*cX+w_i].startY = cY - components[h_i*cX+w_i].startY;
			// printf("count:%d\n", *count);
			int index = atomicAdd(count, 1);
            components_real[index] = components[h_i*cX+w_i];
			// printf("count:%d\n", *count);
			// printf("erea:%d\n", components[h_i*cX+w_i].area);
			// printf("sx:%d\n", sx);
			// printf("sy:%d\n", sy);
			// printf("ex:%d\n", ex);
			// printf("ey:%d\n", ey);
		}
	}

}

// 从label中获取box的坐标
void get_box_from_label(unsigned char *d_image, unsigned int *d_labels, ConnectedComponent *components, ConnectedComponent *components_real, ConnectedComponent *components_real_h, int *count, int *count_h, unsigned int X, unsigned int Y){

    int GRIDDIMX = int((X + BLOCKDIMX - 1) / BLOCKDIMX);
    int GRIDDIMY = int((Y + BLOCKDIMY - 1) / BLOCKDIMY);
    dim3 grid_size(GRIDDIMX, GRIDDIMY, 1);
    dim3 block_size(BLOCKDIMX, BLOCKDIMY, 1);
    cudaMemset(components, 0, sizeof(ConnectedComponent)*X*Y);
    cudaDeviceSynchronize();
    
    get_label_area <<< grid_size,block_size >>> (d_labels, d_image, components, Y, X);
    cudaDeviceSynchronize();

    cudaMemset(count, 0, sizeof(int));
    cudaMemset(components_real, 0, sizeof(ConnectedComponent)*X*Y);
    process_label <<< grid_size,block_size >>> (components, components_real, count, Y, X);
    cudaDeviceSynchronize();
    // Check for any errors
    // checkCUDAErrors();
    
    // 将box数量传输到CPU
    cudaMemcpy(count_h, count, sizeof(int), cudaMemcpyDeviceToHost);

    // 将box相关信息传输到CPU
    cudaMemcpy(components_real_h, components_real, (*count_h) * sizeof(ConnectedComponent), cudaMemcpyDeviceToHost);
}

#if DEBUG 
int main(){
    int width;
    int height;
    width = 3840;
    height = 2160;
    int scale_width, scale_height;
    scale_width = width / 2;
    scale_height = height / 2; 
    int p_count = width * height;
    printf("%d\n", width);

    getpad(scale_width, scale_height);
    
    printf("padding on width: %d\n", padw);
    printf("padding on height: %d\n", padh);

    int new_height = scale_height + padh;
    int new_width = scale_width + padw;

    cudaStream_t streams[3];
    for (int c = 0; c < 3; c++) {
        cudaStreamCreate(&(streams[c]));
    }
    // 开辟内存和显存地址
    uint8_t *src_yuv;
    src_yuv = (uint8_t*) malloc(p_count*sizeof(uint8_t)*3/2);

    uint8_t *src_yuv_i_d;
    cudaMalloc(&src_yuv_i_d, p_count*sizeof(uint8_t)*3/2);

    float *src_yuv_f_d;
    cudaMalloc(&src_yuv_f_d, p_count*sizeof(float)*3);

    half *sdr_rgb;
    cudaMalloc(&sdr_rgb, p_count*sizeof(half)*3);

    half *bilinear_out;
    cudaMalloc(&bilinear_out, p_count/4*sizeof(half)*3);
    float *bilinear_out_f;
    cudaMalloc(&bilinear_out_f, p_count/4*sizeof(float)*3);

    float *bilinear_out_f_h = (float*)malloc(p_count/4*sizeof(float)*3);

    half* pad_out;
    cudaMalloc(&pad_out, new_height*new_width*sizeof(half)*3);
    float* pad_out_f;
    cudaMalloc(&pad_out_f, new_height*new_width*sizeof(float)*3);
    float *pad_out_f_h = (float*)malloc(new_height*new_width*sizeof(float)*3);

    // 读取sdr yuv
    initbuf("./4k-B_fengqiluoyang_17min_27min_toufa_nv12.yuv", src_yuv, sizeof(uint8_t)*p_count*3/2);
    // 将sdr yuv数据传到cuda
    cudaMemcpyAsync(src_yuv_i_d, src_yuv, p_count*sizeof(uint8_t), cudaMemcpyHostToDevice, streams[0]);
    cudaDeviceSynchronize();
    // 将sdr yuv数据归一化成[0,1]的float, 同时420p转换成444p
    nv12_yConvertTo(src_yuv_f_d, src_yuv_i_d, 1/219.0f, 16.0f, width, height, width);
    cudaMemcpyAsync(src_yuv_i_d+p_count, src_yuv+p_count, p_count*sizeof(uint8_t)/2, cudaMemcpyHostToDevice, streams[1]);
    nv12_uvConvertTo(src_yuv_f_d+p_count, src_yuv_f_d+2*p_count, src_yuv_i_d+p_count, 1/224.0f, 16.0f, width, height, width/2, height/2, width);
    yuv444p2rgb_709(src_yuv_f_d, sdr_rgb, height, width);
    cudaDeviceSynchronize();
    
    doscale(streams[2], 1, sdr_rgb, bilinear_out, height, width, scale_height, scale_width);
    cudaDeviceSynchronize();

    pad(bilinear_out, pad_out, scale_height, scale_width, new_height, new_width);
    cudaDeviceSynchronize();
    half2float(pad_out, pad_out_f, new_height, new_width);
    cudaDeviceSynchronize();
    cudaMemcpy(pad_out_f_h, pad_out_f, new_height*new_width*sizeof(float)*3, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    dumpbin("pad.rgb", pad_out_f_h, new_height*new_width*sizeof(float)*3);

    // half2float(bilinear_out, bilinear_out_f, scale_height, scale_width);
    // cudaDeviceSynchronize();
    // cudaMemcpy(bilinear_out_f_h, bilinear_out_f, p_count/4*sizeof(float)*3, cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // dumpbin("./resize.rgb", bilinear_out_f_h, p_count/4*sizeof(float)*3);
    
}
#endif