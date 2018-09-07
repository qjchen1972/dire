#pragma once
#ifndef _PROC_GPU__H
#define _PROC_GPU__H

#include "const.h"
#include "base_gpu.h"
#include "active_gpu.h"
#include "batchnorm_gpu.h"
#include "conv_gpu.h"
#include "end_gpu.h"
#include "grating_gpu.h"
#include "layer_gpu.h"
#include "samp_gpu.h"
#include "scale_gpu.h"
#include "move_gpu.h"
#include "limit_gpu.h"
#include "linearmapping_gpu.h"

#ifdef GPU

//用于所有元素之和一般函数
__device__   GTYPE gpu_sum(gpu_t st, unsigned int pos_x, unsigned int pos_y)
{
	return st.result[st.start + pos_y*st.len+ pos_x];
}

__device__   GTYPE proc_fn(int fn_type, gpu_t st, unsigned int pos_x, unsigned int pos_y)
{
	switch (fn_type)
	{
	case FN_GPU_SUM:
		return gpu_sum(st, pos_x,pos_y);
	case FN_ACTIVE_FW_SUM:
		return active_fw_sum(st, pos_x, pos_y);
	case FN_ACTIVE_FW:
		return active_fw(st, pos_x, pos_y);
	case FN_ACTIVE_BW_SUM:
		return active_bw_sum(st, pos_x, pos_y);
	case FN_ACTIVE_BW:
		return active_bw(st, pos_x, pos_y);
	case FN_ACTIVE_UPDATE_SUM:
		return active_update_sum(st, pos_x, pos_y);
	case FN_BN_FW_EX_SUM:
		return bn_fw_ex_sum(st, pos_x, pos_y);
	case FN_BN_FW_VAR_SUM:
		return bn_fw_var_sum(st, pos_x, pos_y);
	case FN_BN_FW:
		return bn_fw(st, pos_x, pos_y);
	case FN_BN_BW_EX_SUM:
		return bn_bw_ex_sum(st, pos_x, pos_y);
	case FN_BN_BW_EXMULTI_SUM:
		return bn_bw_exmulti_sum(st, pos_x, pos_y);
	case FN_BN_BW:
		return bn_bw(st, pos_x, pos_y);
	case FN_CONV_FW:
		return conv_fw(st, pos_x, pos_y);
	case FN_CONV_BW:
		return conv_bw(st, pos_x, pos_y);
	case FN_CONV_UPDATE:
		return conv_update(st, pos_x, pos_y);
	case FN_CONV_UPDATE_SUM:
		return conv_update_sum(st, pos_x, pos_y);
	case FN_END_BW:
		return end_bw(st, pos_x, pos_y);
	case FN_END_COST_SUM:
		return end_cost_sum(st, pos_x, pos_y);
	case FN_GRATING_FW:
		return grating_fw(st, pos_x, pos_y);
	case FN_GRATING_BW:
		return grating_bw(st, pos_x, pos_y);
	case FN_RESNET_LAYER_ADD:
		return resnet_layer_add(st, pos_x, pos_y);
	case FN_SAMP_FW:
		return samp_fw(st, pos_x, pos_y);
	case FN_SAMP_BW:
		return samp_bw(st, pos_x, pos_y);
	case FN_SAMP_UPDATE_B_SUM:
		return samp_update_b_sum(st, pos_x, pos_y);
	case FN_SAMP_UPDATE_WEIGHT_SUM:
		return samp_update_weight_sum(st, pos_x, pos_y);
	case FN_SCALE_FW:
		return scale_fw(st, pos_x, pos_y);
	case FN_SCALE_BW:
		return scale_bw(st, pos_x, pos_y);
	case FN_SCALE_UPDATE_BDA_SUM:
		return scale_update_bda_sum(st, pos_x, pos_y);
	case FN_SCALE_UPDATE_GEM_SUM:
		return scale_update_gem_sum(st, pos_x, pos_y);
	case FN_MOVE_FW_MAX:
		return move_fw_max(st, pos_x, pos_y);
	case FN_MOVE_FW:
		return move_fw(st, pos_x, pos_y);
	case FN_MOVE_BW:
		return move_bw(st, pos_x, pos_y);
	case FN_LIMIT_FW:
		return limit_fw(st, pos_x, pos_y);
	case FN_LIMIT_BW:
		return limit_bw(st, pos_x, pos_y);
	case FN_DENSENET_LAYER_FW_INPUT:
		return densenet_layer_fw_input(st, pos_x, pos_y);
	case FN_DENSENET_LAYER_BW_INPUT:
		return densenet_layer_bw_input(st, pos_x, pos_y);
	case FN_LINEARMAPPING_FW:
		return linearmapping_fw(st, pos_x, pos_y);
	case FN_LINEARMAPPING_BW:
		return linearmapping_bw(st, pos_x, pos_y);
	case FN_LINEARMAPPING_UPDATE:
		return linearmapping_update(st, pos_x, pos_y);
	default:
		break;
	}
	return 0;
}

//参考斯坦福的课程的代码
//目的是为了计算数组input从start开始的长度为len的子数组的总和
//首先计算每一个线程块的总和
__global__ void block_sum(gpu_t st, int fn_type)
{
	__shared__  GTYPE sdata[GPU_BLOCK];

	//这儿是假设block不超过2^32,若是以后真超过，需要修改代码
	unsigned int  bx = blockIdx.y*gridDim.x +blockIdx.x;
	unsigned int tx = threadIdx.x;
    unsigned int  blk_num = st.len / blockDim.x;
	if (st.len % blockDim.x != 0) blk_num += 1;
	unsigned int  batch_x = bx / blk_num;
	if (batch_x >= st.sum_result_count) return;
	unsigned int  batch_y = bx % blk_num;
	unsigned  int pos_x = batch_y*blockDim.x + tx;

	//这个地方大于st.len不能返回，因为下面使用了blockDim.x进行2分
	if (pos_x >= st.len) 
		sdata[tx] = 0;
	else
	{		
		sdata[tx] = proc_fn(fn_type, st, pos_x, batch_x);
	}
	__syncthreads();//等待所有线程把自己负责的元素载入到共享内存

	int offset ,real_offset;
	for ( offset = blockDim.x / 2 + blockDim.x % 2, real_offset = blockDim.x;
		offset > 0;
		real_offset = offset, offset = (offset == 1 ? 0 : offset / 2 + offset % 2))
	{
		if (threadIdx.x < offset)//控制只有某些线程才进行操作。
		{			
			if (threadIdx.x + offset < real_offset)
			{
				sdata[threadIdx.x] += sdata[threadIdx.x + offset];
			}
		}
		// wait until all threads in the block have
		__syncthreads();
	}
	// 每个块的线程0负责存放块内求和的结果
	if (tx == 0)
	{
		if (!st.istemp)
		{
			st.sum_result[bx] = sdata[tx];
		}
		else		
			st.per_block_result[bx] = sdata[tx];			
	}
}

//数组计算
__global__ void cal(gpu_t st, int fn_type)
{
	//这儿是假设block不超过2^32,若是以后真超过，需要修改代码
	unsigned int bx = blockIdx.y*gridDim.x + blockIdx.x;
	int tx = threadIdx.x;
	unsigned int pos = bx*blockDim.x + tx;
	if (pos >= st.len) return;
	proc_fn(fn_type, st, pos,0);
}
#endif
#endif
