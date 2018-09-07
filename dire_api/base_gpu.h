#pragma once
#ifndef _BASE_GPU__H
#define _BASE_GPU__H

#include "const.h"


#ifdef GPU
#include<cuda.h>

//gpu�߳̿�����ֵ
#define  GPU_BLOCK  64
//���������߳̿���
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
	unsigned short  kernel_row; //����˵���
	unsigned short  kernel_col; //����˵���
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

//ÿһ�㲻������ݽṹ
struct  __align__(ALIGN_SIZE) gpu_layer_t
{
	// ��������
	unsigned short  fw_row; //ǰ���������
	unsigned short  fw_col; //ǰ���������
	unsigned short  fw_real_row; //ǰ��ʵ���������
	unsigned short  fw_real_col;	//ǰ��ʵ���������
	unsigned short  fw_node_num; // �ڵ���
	unsigned int  fw_batch_size; //����batch�ĳߴ�
	unsigned int  fw_node_size;

	unsigned short  bw_row; //�����������
	unsigned short  bw_col; //�����������
	unsigned short  bw_node_num; // �ڵ���
	unsigned int  bw_batch_size; //����batch�ĳߴ�
	unsigned int  bw_node_size;
	union layer_type type;
};


struct  __align__(ALIGN_SIZE)  gpu_t
{	
	GTYPE  *result = nullptr;
	unsigned int result_layer = 0;

	//��������Ľ��
	unsigned long long *in = nullptr;
	unsigned int *in_layer = nullptr;
	unsigned int *in_pos = nullptr;	

	//Ŀǰֻ����4���������飬��������
	GTYPE  *in1 = nullptr;
	unsigned int in1_layer = 0;

	GTYPE  *in2 = nullptr;
	unsigned int in2_layer = 0;

	GTYPE  *in3 = nullptr;
	unsigned int in3_layer = 0;

	GTYPE  *in4 = nullptr;
	unsigned int in4_layer = 0;

	//����һЩ��������
	unsigned int *in5 = nullptr;

	//��������ķ�Χ
	unsigned int start = 0;
	unsigned int len = 0;

	//�����ۼƣ�+ - * /��ʱ����ʱ���飬֮������ʱ����ҲҪ���ݣ�����cudaMalloc��cudamemcpy�Ƚ������������ȷ����
	GTYPE  *per_block_result = nullptr;
	//�ۼƽ�������
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


//��������
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