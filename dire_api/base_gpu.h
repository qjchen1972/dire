#pragma once
#ifndef _BASE_GPU__H
#define _BASE_GPU__H

#include "const.h"


#ifdef GPU
#include<cuda.h>

//gpu线程块的最大值
#define  GPU_BLOCK  64
//定义最大的线程块数
#define  GPU_GRID   0x7fffffff

#define ALIGN_SIZE 8

#pragma pack(push,ALIGN_SIZE)

struct __align__(ALIGN_SIZE) active_t
{
	unsigned char  active_type;
};

struct __align__(ALIGN_SIZE) end_t
{
	unsigned char cost_type;
};

struct __align__(ALIGN_SIZE) grating_t
{
	unsigned char grating_type;
};

struct __align__(ALIGN_SIZE) samp_t
{
	unsigned short  pool_row;
	unsigned short  pool_col;
	unsigned char  pool_type;
	unsigned char  top_padding;
	unsigned char  bottom_padding;
	unsigned char  left_padding;
	unsigned char  right_padding;
	unsigned char  row_stride;
	unsigned char  col_stride;
};

struct __align__(ALIGN_SIZE) conv_t
{
	unsigned short  kernel_row; //卷积核的行
	unsigned short  kernel_col; //卷积核的列
	unsigned char  top_padding;
	unsigned char  bottom_padding;
	unsigned char  left_padding;
	unsigned char  right_padding;
	unsigned char  row_stride;
	unsigned char  col_stride;
};

union layer_type
{
	active_t  active;
	end_t  end;
	grating_t  grating;
	samp_t  samp;
	conv_t  conv;
};

//每一层不变的数据结构
struct  __align__(ALIGN_SIZE) gpu_layer_t
{
	// 个性数据
	unsigned short  fw_row; //前向输出的行
	unsigned short  fw_col; //前向输出的列
	unsigned short  fw_real_row; //前向实际输出的行
	unsigned short  fw_real_col;	//前向实际输出的列
	unsigned short  fw_node_num; // 节点数
	unsigned int  fw_batch_size; //单个batch的尺寸
	unsigned int  fw_node_size;

	unsigned short  bw_row; //后向输出的行
	unsigned short  bw_col; //后向输出的列
	unsigned short  bw_node_num; // 节点数
	unsigned int  bw_batch_size; //单个batch的尺寸
	unsigned int  bw_node_size;
	union layer_type type;
};


struct  __align__(ALIGN_SIZE)  gpu_t
{	
	GTYPE  *result = nullptr;
	unsigned int result_layer = 0;

	//用于输出的结果
	unsigned long long *in = nullptr;
	unsigned int *in_layer = nullptr;
	unsigned int *in_pos = nullptr;	

	//目前只允许4个输入数组，用来计算
	GTYPE  *in1 = nullptr;
	unsigned int in1_layer = 0;

	GTYPE  *in2 = nullptr;
	unsigned int in2_layer = 0;

	GTYPE  *in3 = nullptr;
	unsigned int in3_layer = 0;

	GTYPE  *in4 = nullptr;
	unsigned int in4_layer = 0;

	//用于一些整形数据
	unsigned int *in5 = nullptr;

	//计算数组的范围
	unsigned int start = 0;
	unsigned int len = 0;

	//计算累计（+ - * /）时的临时数组，之所以临时数组也要传递，在于cudaMalloc和cudamemcpy比较慢，所以事先分配好
	GTYPE  *per_block_result = nullptr;
	//累计结果的输出
	GTYPE  *sum_result = nullptr;
	unsigned int  sum_result_count = 1;
	int istemp = 0;
	unsigned int  batchsize = 0;
};

#pragma pack(pop)

#if MAX_LAYER_NUM < 1171
__constant__ gpu_layer_t g_layer[MAX_LAYER_NUM];
#else
__device__ gpu_layer_t g_layer[MAX_LAYER_NUM];
#endif
gpu_layer_t  g_cpu_layer[MAX_LAYER_NUM];


//函数类型
#define FN_GPU_SUM  0
#define FN_ACTIVE_FW_SUM  1
#define FN_ACTIVE_FW  2
#define FN_ACTIVE_BW_SUM  3
#define FN_ACTIVE_BW  4
#define FN_ACTIVE_UPDATE_SUM  5
#define FN_BN_FW_EX_SUM  6
#define FN_BN_FW_VAR_SUM  7
#define FN_BN_FW_CAL_TOTAL  8
#define FN_BN_FW_TEST  9
#define FN_BN_FW  10
#define FN_BN_BW_EX_SUM  11
#define FN_BN_BW_EXMULTI_SUM  12
#define FN_BN_BW  13
#define FN_CONV_FW  14
#define FN_CONV_BW  15
#define FN_CONV_UPDATE  16
#define FN_CONV_UPDATE_SUM  17
#define FN_END_BW  18
#define FN_END_COST_SUM  19
#define FN_RESNET_LAYER_ADD  21
#define FN_SAMP_FW  22
#define FN_SAMP_BW  23
#define FN_SAMP_UPDATE_B_SUM  24
#define FN_SAMP_UPDATE_WEIGHT_SUM  25
#define FN_SCALE_FW  26
#define FN_SCALE_BW  27
#define FN_SCALE_UPDATE_BDA_SUM  28
#define FN_SCALE_UPDATE_GEM_SUM  29
#define FN_UNSCALED_FW_MAX  32
#define FN_UNSCALED_FW  33
#define FN_UNSCALED_BW_SUM  34
#define FN_UNSCALED_BW  35
#define FN_MOVE_FW_MAX  36
#define FN_MOVE_FW  37
#define FN_MOVE_BW  39
#define FN_LIMIT_FW  40
#define FN_LIMIT_BW  41
#define FN_GRATING_FW  42
#define FN_GRATING_BW  43
#define FN_DENSENET_LAYER_FW_INPUT 44
#define FN_DENSENET_LAYER_BW_INPUT 45
#define  FN_LINEARMAPPING_FW 46
#define  FN_LINEARMAPPING_BW 47
#define  FN_LINEARMAPPING_UPDATE 48

static void HandleError(cudaError_t err, const char *file, int line) 
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void init_layer_const()
{
	HANDLE_ERROR(cudaMemcpyToSymbol(g_layer, g_cpu_layer, sizeof(gpu_layer_t)*MAX_LAYER_NUM));
}
#endif
#endif