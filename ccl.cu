#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
//#include <opencv2/opencv.hpp>
#define THREADS_X 32
#define THREADS_Y 32
// Block Size (Block X should always be 32 for __shfl to work correctly)
#define BLOCK_X 32
#define BLOCK_Y 4
#define REDUCE_DOT_LIMIT 20

// Remove if BLOCK dimensions are not a power-of-two
#define BLOCK_IS_POW2
// Image Size (Device Constants)
__constant__ unsigned int cX, cY, cXY, cS;

// Shift (Constants)
__constant__ unsigned int sX, sY, sZ;
__constant__ unsigned int mX, mY, mZ;
__constant__ unsigned int mb0, mb1;

__device__ __forceinline__ unsigned int LaneId() {
	unsigned int ret;
	asm("mov.u32 %0, %%laneid;" : "=r"(ret));
	return ret;
}

// ---------- Find the root of a chain ----------
__device__ __inline__ unsigned int find_root(unsigned int *labels,
		unsigned int label) {
	// Resolve Label
	unsigned int next = labels[label];

	// Follow chain
	while (label != next) {
		// Move to next
		label = next;
		next = labels[label];
	}

	// Return label
	return label;
}

// ---------- Label Reduction ----------
__device__ __inline__ unsigned int
reduction(unsigned int *g_labels, unsigned int label1, unsigned int label2) {
	// Get next labels
	unsigned int next1 = (label1 != label2) ? g_labels[label1] : 0;
	unsigned int next2 = (label1 != label2) ? g_labels[label2] : 0;

	// Find label1
	while ((label1 != label2) && (label1 != next1)) {
		// Adopt label
		label1 = next1;

		// Fetch next label
		next1 = g_labels[label1];
	}

	// Find label2
	while ((label1 != label2) && (label2 != next2)) {
		// Adopt label
		label2 = next2;

		// Fetch next label
		next2 = g_labels[label2];
	}

	unsigned int label3;
	// While Labels are different
	while (label1 != label2) {
		// Label 2 should be smallest
		if (label1 < label2) {
			// Swap Labels
			label1 = label1 ^ label2;
			label2 = label1 ^ label2;
			label1 = label1 ^ label2;
		}

		// AtomicMin label1 to label2
		label3 = atomicMin(&g_labels[label1], label2);
		label1 = (label1 == label3) ? label2 : label3;
	}

	// Return label1
	return label1;
}

// Resolve Kernel
__global__ void resolve_labels(unsigned int *g_labels) {
	unsigned int b = blockIdx.x / mb0;
	unsigned int* dout = g_labels + b * cXY;
	// Calculate index
	const unsigned int id = ((blockIdx.y * blockDim.y) + threadIdx.y) * cX +
		(((blockIdx.x - b*mb0)* blockDim.x) + threadIdx.x);

	// Check Thread Range
	if (id < cXY) {
		// Resolve Label
		dout[id] = find_root(dout, dout[id]);
	}
}

// Playne-Equivalence Block-Label method
__global__ void block_label(unsigned int *g_labels, unsigned char *g_image) {
	// Shared Memory Label Cache
	extern __shared__ unsigned int s_labels[];

	// Calculate the index inside the grid
	unsigned int b = blockIdx.x / mb0;
	unsigned char *din = g_image + b * cXY;
	unsigned int* dout = g_labels + b * cXY;
	const unsigned int ix = (((blockIdx.x - b * mb0)* blockDim.x) + threadIdx.x);
	const unsigned int iy = ((blockIdx.y * blockDim.y) + threadIdx.y);

	// Check Thread Range
	if ((ix < cX) && (iy < cY)) {
		// Calculate the index inside the block
		const unsigned int bx = threadIdx.x;
		const unsigned int by = threadIdx.y;

		// ----------------------------------------
		// Global Memory - Neighbour Connections
		// ----------------------------------------
		// Load pixels
		const unsigned char pyx = din[iy * cX + ix];

		// Load neighbour from global memory
		const unsigned char pym1x = (by > 0) ? din[(iy - 1) * cX + ix] : 0;

		// Shuffle Pixels
		const unsigned char pyxm1 = __shfl_up_sync(__activemask(), pyx, 1);
		const unsigned char pym1xm1 = __shfl_up_sync(__activemask(), pym1x, 1);

		// Neighbour Connections
		const bool nym1x = (by > 0) ? (pyx == pym1x) : false;
		const bool nyxm1 = (bx > 0) ? (pyx == pyxm1) : false;
		const bool nym1xm1 = (by > 0 && bx > 0) ? (pyx == pym1xm1) : false;

		// Label
		unsigned int label1;

		// ---------- Initialisation ----------
		label1 = (nyxm1) ? by * blockDim.x + (bx - 1) : by * blockDim.x + bx;
		label1 = (nym1x) ? (by - 1) * blockDim.x + bx : label1;

		// Write label to shared memory
		s_labels[by * blockDim.x + bx] = label1;

		// Synchronise Threads
		__syncthreads();

		// ---------- Analysis ----------
		// Resolve Label
		s_labels[by * blockDim.x + bx] = find_root(s_labels, label1);

		// Synchronise Threads
		__syncthreads();

		// ---------- Reduction ----------
		// Check critical
		if (nym1x && nyxm1 && !nym1xm1) {
			// Get neighbouring label
			unsigned int label2 = s_labels[by * blockDim.x + bx - 1];

			// Reduction
			label1 = reduction(s_labels, label1, label2);
		}

		// Synchronise Threads
		__syncthreads();

		// ---------- Analysis ----------
		// Resolve Label
		label1 = find_root(s_labels, label1);

#ifdef BLOCK_IS_POW2
		// Extract label components
		const unsigned int lx = label1 & mX;
		const unsigned int ly = (label1 >> sY) & mY;
#else
		// Extract label components
		const unsigned int lx = label1 % blockDim.x;
		const unsigned int ly = (label1 / blockDim.x) % blockDim.y;
#endif

		// Write to Global
		dout[iy * cX + ix] = ((blockIdx.y * blockDim.y) + ly) * cX +
			(((blockIdx.x - b*mb0)* blockDim.x) + lx);
	}
}

// X - Reduction
__global__ void x_label_reduction(unsigned int *g_labels,
		unsigned char *g_image) {
	unsigned int b = blockIdx.x / mb1;
	// Calculate Index
	unsigned int ix =
		((blockIdx.y * blockDim.y) + threadIdx.y) * BLOCK_X + BLOCK_X;
	unsigned int iy = (((blockIdx.x - b*mb1)* blockDim.x) + threadIdx.x);
	unsigned char *din = g_image + b * cXY;
	unsigned int* dout = g_labels + b * cXY;

	// Check Range
	if (ix < cX && iy < cY) {
		// Get image and label values
		const unsigned char pyx = din[iy * cX + ix];

		// Neighbour Values
		const unsigned char pyxm1 = din[iy * cX + ix - 1];

		// Edge of block flag
#ifdef BLOCK_IS_POW2
		const bool thread_y = (iy & mY) == 0;
#else
		const bool thread_y = (iy % BLOCK_Y) == 0;
#endif

		// Fetch Neighbours
		const unsigned char pym1x = __shfl_up_sync(__activemask(), pyx, 1);
		const unsigned char pym1xm1 = __shfl_up_sync(__activemask(), pyxm1, 1);

		// If connected to left neighbour
		if ((pyx == pyxm1) && (thread_y || (pyx != pym1x) || (pyx != pym1xm1))) {
			// Get Labels
			unsigned int label1 = dout[iy * cX + ix];
			unsigned int label2 = dout[iy * cX + ix - 1];

			// Reduction
			reduction(dout, label1, label2);
		}
	}
}

// Y - Reduction
__global__ void y_label_reduction(unsigned int *g_labels,
		unsigned char *g_image) {
	unsigned int b = blockIdx.x / mb0;
	// Calculate Index
	unsigned int ix = (((blockIdx.x - b*mb0)* blockDim.x) + threadIdx.x);
	unsigned int iy =
		((blockIdx.y * blockDim.y) + threadIdx.y) * BLOCK_Y + BLOCK_Y;
	unsigned char *din = g_image + b * cXY;
	unsigned int* dout = g_labels + b * cXY;

	// Check Range
	if (ix < cX && iy < cY) {
		// Get image and label values
		const unsigned char pyx = din[iy * cX + ix];
		const unsigned char pym1x = din[(iy - 1) * cX + ix];

		// Neighbour Connections
		const unsigned char pyxm1 = __shfl_up_sync(__activemask(), pyx, 1);
		const unsigned char pym1xm1 = __shfl_up_sync(__activemask(), pym1x, 1);

		// If connected to neighbour
		if ((pyx == pym1x) &&
				((threadIdx.x == 0) || (pyx != pyxm1) || (pyx != pym1xm1))) {
			// Get labels
			unsigned int label1 = dout[iy * cX + ix];
			unsigned int label2 = dout[(iy - 1) * cX + ix];

			// Reduction
			reduction(dout, label1, label2);
		}
	}
}

__device__ int reflect101(int index, int endIndex) {
	return abs(endIndex - abs(endIndex - index));
}

__device__ unsigned char load2ShrdMemRGB(uchar3 *in, int gx, int gy, int rows,
		int cols) {
	int idx = reflect101(gx, rows - 1) * cols + reflect101(gy, cols - 1);
	unsigned int r = in[idx].z;
	unsigned int g = in[idx].y;
	unsigned int b = in[idx].x;
	return (r * 76 + g * 150 + b * 30) >> 8;
}

__device__ void load2ShrdMemMorph(unsigned char *shm,
		unsigned char *in, int lx, int ly,
		int shrdStride, int gx, int gy,
		int imgRows, int imgCols,
		bool isDilation) {
	unsigned char val;
	if (isDilation)
		val = 0;
	else
		val = 1;

	if (gx >= 0 && gx < imgCols && gy >= 0 && gy < imgRows) {
		val = in[gx + gy * imgCols];
	}

	shm[ly * shrdStride + lx] = val;
}

__global__ void scharr3x3(uchar3 *dataIn, unsigned char *dataOut,
		short int imgRows, short int imgCols, short int nBBS0,
		short int nBBS1, unsigned int batchsize) {
	__shared__ unsigned char shrdMem[THREADS_X + 2][THREADS_Y + 2];

	// calculate necessary offset and window parameters
	const int radius = 1;
	const int padding = 2 * radius;
	const int shrdLen = blockDim.x + padding;

	// batch offsets
	unsigned int b = blockIdx.x / nBBS0;
	uchar3 *iptr = dataIn + b * batchsize;
	unsigned char *dptr = dataOut + b * batchsize;

	// local neighborhood indices
	int lx = threadIdx.x;
	int ly = threadIdx.y;

	// global indices
	int gx = THREADS_X * (blockIdx.x - b * nBBS0) + lx;
	int gy = THREADS_Y * (blockIdx.y) + ly;

	for (int b = ly, gy2 = gy; b < shrdLen; b += blockDim.y, gy2 += blockDim.y) {
		for (int a = lx, gx2 = gx; a < shrdLen;
				a += blockDim.x, gx2 += blockDim.x) {
			shrdMem[a][b] =
				load2ShrdMemRGB(iptr, gx2 - radius, gy2 - radius, imgRows, imgCols);
		}
	}

	__syncthreads();

	// Only continue if we're at a valid location
	if (gx < imgRows && gy < imgCols) {
		int i = lx + radius;
		int j = ly + radius;
		int _i = i - 1;
		int i_ = i + 1;
		int _j = j - 1;
		int j_ = j + 1;

		float NW = shrdMem[_i][_j];
		float SW = shrdMem[i_][_j];
		float NE = shrdMem[_i][j_];
		float SE = shrdMem[i_][j_];

		float t1 = shrdMem[_i][j];
		float t2 = shrdMem[i_][j];

		// FIXME: use tensor core to accelerate further
		float dx = (NW - SW + NE - SE) * 3.0 + (t1 - t2) * 10.0;

		t1 = shrdMem[i][_j];
		t2 = shrdMem[i][j_];
		float dy = (NW + SW - NE - SE) * 3.0 + (t1 - t2) * 10.0;

		if (dx < 0)
			dx = -dx;
		if (dy < 0)
			dy = -dy;

		float out = (dx + dy) / 2;
		if (out > 100.0)
			dptr[gx * imgCols + gy] = 255;
		else
			dptr[gx * imgCols + gy] = 0;
	}
}

__global__ void hist(unsigned int *d_labels, unsigned int *mask, int limit) {
	unsigned int b = blockIdx.x / (cY*cS);
	unsigned int *din = d_labels+ b * cXY;
	unsigned int* dout = mask+ b * cXY;
	int id = (blockIdx.x- b*cY*cS) * blockDim.x + threadIdx.x;
	unsigned int label = din[id];

	if (dout[label] < limit)
		atomicAdd(&dout[label], 1);
}

__global__ void reducedot(unsigned char *sobel, unsigned int *d_labels,
		unsigned int *mask, unsigned char *reduce, int limit) {
	unsigned int b = blockIdx.x / (cY*cS);
	unsigned int *d0 = d_labels+ b * cXY;
	unsigned int* d1 = mask+ b * cXY;
	unsigned char *s0 = sobel + b*cXY;
	unsigned char *s1 = reduce + b*cXY;
	int id = (blockIdx.x - b*cY*cS)* blockDim.x + threadIdx.x;
	if (s0[id] > 0 && d1[d0[id]] < limit) {
		s0[id] = 0;
		s1[id] = 1;
	}
}

__global__ void postprocess(uchar3 *src, uchar3 *qrs, unsigned char *sobel, unsigned char *reduce) {
	unsigned int b = blockIdx.x / (cY*cS);
	unsigned char *s0 = sobel + b*cXY;
	unsigned char *s1 = reduce + b*cXY;
	uchar3 *d0 = src + b * cXY;
	uchar3 *d1 = qrs + b * cXY;
	int id = (blockIdx.x - b*cY*cS)* blockDim.x + threadIdx.x;

	// src + (qrs-src)*mask
	if (s0[id] == 0) {
		if (s1[id] == 0) {
			d0[id].x = (d0[id].x+d1[id].x)/2;
			d0[id].y = (d0[id].y+d1[id].y)/2;
			d0[id].z = (d0[id].z+d1[id].z)/2;
		}
	} else
		d0[id] = d1[id];
}

void scharr(uchar3 *dataIn, unsigned char *dataOut, short int imgRows,
		short int imgCols, int batch) {
	short int nBBS0, nBBS1;
	unsigned int batchsize;
	nBBS0 = (imgRows + THREADS_X - 1) / THREADS_X;
	nBBS1 = (imgCols + THREADS_Y - 1) / THREADS_Y;
	batchsize = imgRows * imgCols;

	dim3 threadsPerBlock(THREADS_X, THREADS_Y);
	dim3 blockPerGrid(nBBS0 * batch, nBBS1);
	scharr3x3<<<blockPerGrid, threadsPerBlock>>>(
			dataIn, dataOut, imgRows, imgCols, nBBS0, nBBS1, batchsize);
}

__constant__ unsigned char cFilter[9] = {
	1, 1, 1, 1, 1, 1, 1, 1, 1,
};
__global__ void morph(unsigned char *dataOut, unsigned char *dataIn,
		int imgCols, int imgRows, int nBBS0, int nBBS1,
		int batchsize, bool isDilation) {
	__shared__ unsigned char shrdMem[(THREADS_X + 3)*(THREADS_Y + 2)];
	int windLen = 3;

	// calculate necessary offset and window parameters
	const int halo = windLen / 2;
	const int padding = (windLen % 2 == 0 ? (windLen - 1) : (2 * (windLen / 2)));
	const int shrdLen = blockDim.x + padding + 1;
	const int shrdLen1 = blockDim.y + padding;
	// gfor batch offsets
	unsigned b = blockIdx.x / nBBS0;
	unsigned char *iptr = dataIn + b * batchsize;
	unsigned char *optr = dataOut + b * batchsize;

	const int lx = threadIdx.x;
	const int ly = threadIdx.y;

	// global indices
	const int gx = blockDim.x * (blockIdx.x - b * nBBS0) + lx;
	const int gy = blockDim.y * (blockIdx.y) + ly;

	// pull image to local memory
	for (int b = ly, gy2 = gy; b < shrdLen1; b += blockDim.y, gy2 += blockDim.y) {
		// move row_set get_local_size(1) along coloumns
		for (int a = lx, gx2 = gx; a < shrdLen;
				a += blockDim.x, gx2 += blockDim.x) {
			load2ShrdMemMorph(shrdMem, iptr, a, b, shrdLen, gx2 - halo, gy2 - halo,
					imgRows, imgCols, isDilation);
		}
	}

	int i = lx + halo;
	int j = ly + halo;

	__syncthreads();
	const unsigned char *d_filt = (const unsigned char *)cFilter;
	unsigned char acc = isDilation ? 0 : 255;
	#pragma unroll
	for (int wj = 0; wj < windLen; ++wj) {
		int joff = wj * windLen;
		int w_joff = (j + wj - halo) * shrdLen;
		#pragma unroll
		for (int wi = 0; wi < windLen; ++wi) {
			if (d_filt[joff + wi] > 0) {
				unsigned char cur = shrdMem[w_joff + (i + wi - halo)];
				if (isDilation)
					acc = max(acc, cur);
				else
					acc = min(acc, cur);
			}
		}
	}

	if (gx < imgCols && gy < imgRows) {
		optr[gy * imgCols + gx] = acc;
	}
}

void morphclose(unsigned char *dataIn, unsigned char *dataOut,
		short int imgRows, short int imgCols, int batch) {
	short int nBBS0, nBBS1;
	unsigned int batchsize;
	nBBS0 = (imgRows + THREADS_X - 1) / THREADS_X;
	nBBS1 = (imgCols + THREADS_Y - 1) / THREADS_Y;
	batchsize = imgRows * imgCols;

	dim3 threadsPerBlock(THREADS_X, THREADS_Y);
	dim3 blockPerGrid(nBBS1 * batch, nBBS0);
	morph<<<blockPerGrid, threadsPerBlock>>>(dataOut, dataIn, imgCols, imgRows,
			nBBS1, nBBS0, batchsize, true);
	morph<<<blockPerGrid, threadsPerBlock>>>(dataIn, dataOut, imgCols, imgRows,
			nBBS1, nBBS0, batchsize, false);
}

static int imgRows=0;
static int imgCols=0;
// static unsigned char *sobel, *reduce;
// static unsigned int *d_labels, *mask;
// static cudaStream_t stream1, stream2, stream3, stream4;
static bool inited = false;
static unsigned int batchsize;

// static void doinit() {
// 	if (inited) return;

// 	inited = true;
// 	cudaStreamCreate(&stream1);
// 	cudaStreamCreate(&stream2);
// 	cudaStreamCreate(&stream3);
// 	cudaStreamCreate(&stream4);
// }

//extern "C"
void ccl(unsigned char* origin, unsigned int *d_labels, int height, int width, int batch, cudaStream_t stream1, cudaStream_t stream2) {
	// unsigned int _batchsize = height*width*batch;
	dim3 block(BLOCK_X, BLOCK_Y);
	int scale = 2;
	if (width > 3840)
        scale = 8;
	else if (width > 1920)
		scale = 4;
     
	if (imgRows != height || imgCols != width) {
		imgRows = height;
		imgCols = width;
		int rsz = imgCols * imgRows;
		unsigned int SX = 0;
		unsigned int SY = (int)(log2((float)block.x));
		unsigned int MX = (block.x - 1);
		unsigned int MY = (block.y - 1);
		unsigned int b0 = ceil(imgCols / (float)BLOCK_X);
		unsigned int b1 = ceil(imgRows / (float)BLOCK_X);

		// doinit();
		// if (inited) {
		// 	cudaFree(mask);
		// 	cudaFree(sobel);
		// 	cudaFree(reduce);
		// 	cudaFree(d_labels);
		// }
		// cudaMalloc(&mask, width * height * sizeof(unsigned int) *batch);
		// cudaMalloc(&sobel, width*height*batch);
		// cudaMalloc(&reduce, width*height*batch);
		// cudaMalloc(&d_labels, imgRows * imgCols * sizeof(unsigned int)*batch);
		// Copy host to device memory
		cudaMemcpyToSymbol(cX, &imgCols, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(cY, &imgRows, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(cS, &scale, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(cXY, &rsz, sizeof(unsigned int), 0,
				cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(sX, &SX, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(sY, &SY, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(mX, &MX, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(mY, &MY, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(mb0, &b0, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(mb1, &b1, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
	}

	dim3 threadsPerBlock(THREADS_X, THREADS_Y);
	dim3 blockPerGrid((imgCols + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(imgRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Create Grid/Block
	dim3 grid(ceil(imgCols / (float)block.x) * batch, ceil(imgRows / (float)block.y));

	// Create Border X/Y Grid
	dim3 border_grid_x(ceil(imgRows / (float)block.x) *batch, ceil(grid.x / (float)block.y));
	dim3 border_grid_y(ceil(imgCols / (float)block.x) * batch, ceil(grid.y / (float)block.y));

	// cudaMemsetAsync(mask, 0, imgCols * imgRows * 4* batch, stream3);
	// cudaMemsetAsync(reduce, 0, imgCols * imgRows * batch, stream4);
	// scharr((uchar3 *)origin, sobel, imgRows, imgCols, batch);
	block_label<<<grid, block, block.x * block.y * sizeof(unsigned int)>>>(
			d_labels, origin);
	cudaDeviceSynchronize();

	y_label_reduction<<<border_grid_y, block, 0, stream1>>>(d_labels, origin);
	x_label_reduction<<<border_grid_x, block, 0, stream2>>>(d_labels, origin);
	cudaDeviceSynchronize();
	resolve_labels<<<grid, block>>>(d_labels);
	cudaDeviceSynchronize();
	
}
