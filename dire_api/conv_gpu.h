#pragma once
#ifndef _CONV_GPU__H
#define _CONV_GPU__H

#include "base_gpu.h"

#ifdef GPU

/*
struct gpu_t st;
st.result = m_fw_output;
st.result_layer = m_gpu_layer;
st.in1 = m_kernel_gpu;
if(m_biasinuse)	st.in2 = m_kernel_b_gpu;
st.in3 = m_fw_total_input;
st.start = 0;
st.len = m_fw_buff_size;
gpu_cal(st, FN_CONV_FW);
*/
__device__   GTYPE conv_fw(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{
	unsigned  int pos = pos_y * st.len + pos_x;

	unsigned int   kernel_block_size = g_layer[st.result_layer].type.conv.kernel_row * g_layer[st.result_layer].type.conv.kernel_col;
	unsigned int   kernel_node_size = g_layer[st.result_layer].bw_node_num* kernel_block_size;
	
	unsigned int   batch_x = pos / g_layer[st.result_layer].fw_batch_size;
	unsigned int   batch_y = pos % g_layer[st.result_layer].fw_batch_size;

	unsigned int   node_x = batch_y / g_layer[st.result_layer].fw_node_size;
	unsigned int   node_y = batch_y % g_layer[st.result_layer].fw_node_size;

	int x = node_y / g_layer[st.result_layer].fw_col;
	int y = node_y % g_layer[st.result_layer].fw_col;

	if (x < g_layer[st.result_layer].type.conv.top_padding || x >= g_layer[st.result_layer].fw_row - g_layer[st.result_layer].type.conv.bottom_padding ||
		y < g_layer[st.result_layer].type.conv.left_padding || y >= g_layer[st.result_layer].fw_col - g_layer[st.result_layer].type.conv.right_padding)
	{
		st.result[pos] = 0;
	}
	else
	{
		int s = (x - g_layer[st.result_layer].type.conv.top_padding) * g_layer[st.result_layer].type.conv.row_stride;
		int t = (y - g_layer[st.result_layer].type.conv.left_padding) * g_layer[st.result_layer].type.conv.col_stride;

		
		if(st.in2)	
			st.result[pos] = st.in2[node_x];
		else
			st.result[pos] = 0;

		
		//kernel 的row pos
		unsigned int    ker_row_pos = node_x*kernel_node_size;
		//input在哪个batch
		unsigned int    input_batch_pos = batch_x * g_layer[st.result_layer].bw_batch_size;
		unsigned int   j;
		for (j = 0; j < kernel_node_size; j++)
		{
			unsigned int   prev_node_x = j / kernel_block_size;
			unsigned int   prev_node_y = j % kernel_block_size;

			int u = prev_node_y / g_layer[st.result_layer].type.conv.kernel_col;
			int v = prev_node_y % g_layer[st.result_layer].type.conv.kernel_col;

			if (s + u < g_layer[st.result_layer].bw_row && t + v < g_layer[st.result_layer].bw_col)
				st.result[pos] += st.in1[ker_row_pos + j] * st.in3[input_batch_pos + prev_node_x * g_layer[st.result_layer].bw_node_size + 
				(s + u)* g_layer[st.result_layer].bw_col + t + v];
		}		
	}
	return st.result[pos];
}

/*
struct gpu_t st;
st.result = m_bw_output;
st.result_layer = m_gpu_layer;
st.in1 = m_kernel_gpu;
st.in2 = m_bw_total_input;
st.start = 0;
st.len = m_bw_buff_size;
gpu_cal(st, FN_CONV_BW);
*/
__device__   GTYPE conv_bw(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{
	unsigned  int pos = pos_y * st.len + pos_x;

	unsigned int   kernel_block_size = g_layer[st.result_layer].type.conv.kernel_row * g_layer[st.result_layer].type.conv.kernel_col;
	unsigned int   kernel_nodecol_size = g_layer[st.result_layer].bw_node_num* kernel_block_size;
	unsigned int   kernel_noderow_size = g_layer[st.result_layer].fw_node_num * kernel_block_size;

	unsigned int   batch_x = pos / g_layer[st.result_layer].bw_batch_size;
	unsigned int   batch_y = pos % g_layer[st.result_layer].bw_batch_size;

	unsigned int   prev_node_x = batch_y / g_layer[st.result_layer].bw_node_size;
	unsigned int   prev_node_y = batch_y % g_layer[st.result_layer].bw_node_size;

	int x = prev_node_y / g_layer[st.result_layer].bw_col;
	int y = prev_node_y % g_layer[st.result_layer].bw_col;

	st.result[pos] = 0;
	//kernel 位置
	unsigned int   ker_pos = prev_node_x*kernel_block_size;
	// bw input 所在的batch
	unsigned int   bw_batch_pos = batch_x * g_layer[st.result_layer].fw_batch_size;
	unsigned int   j;
	for (j = 0; j < kernel_noderow_size; j++)
	{
		unsigned int   node_x = j / kernel_block_size;
		unsigned int   node_y = j % kernel_block_size;

		int u = node_y / g_layer[st.result_layer].type.conv.kernel_col;
		int v = node_y % g_layer[st.result_layer].type.conv.kernel_col;

		int s = (x - u) / g_layer[st.result_layer].type.conv.row_stride + g_layer[st.result_layer].type.conv.top_padding;
		int t = (y - v) / g_layer[st.result_layer].type.conv.col_stride + g_layer[st.result_layer].type.conv.left_padding;

		if (x - u >= 0 && y - v >= 0 && (x - u) % g_layer[st.result_layer].type.conv.row_stride == 0 && (y - v) % g_layer[st.result_layer].type.conv.col_stride == 0 &&
			(x - u) / g_layer[st.result_layer].type.conv.row_stride < g_layer[st.result_layer].fw_real_row && (y - v) / g_layer[st.result_layer].type.conv.col_stride < g_layer[st.result_layer].fw_real_col)
			st.result[pos] += st.in1[node_x*kernel_nodecol_size + ker_pos + node_y] * st.in2[bw_batch_pos + node_x * g_layer[st.result_layer].fw_node_size + s * g_layer[st.result_layer].fw_col + t];
	}
	return st.result[pos];
}

/*
kernel_buff_size = m_node_num*m_prev_layer1->m_node_num*m_kernel_row*m_kernel_col;
struct gpu_t st;
st.result = m_kernel_diff_gpu;
st.result_layer = m_gpu_layer;
st.in1 = m_fw_total_input;
st.in2 = m_bw_total_input;
st.start = 0;
st.len = kernel_buff_size;
st.batchsize = m_batch_num;
gpu_cal(st, FN_CONV_UPDATE);
*/
__device__ GTYPE conv_update(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{	
	unsigned  int pos = pos_y * st.len + pos_x;
	unsigned int   kernel_block_size = g_layer[st.result_layer].type.conv.kernel_row * g_layer[st.result_layer].type.conv.kernel_col;
	unsigned int   kernel_node_size = g_layer[st.result_layer].bw_node_num * kernel_block_size;
	unsigned int   nodes_len = st.batchsize * g_layer[st.result_layer].fw_row * g_layer[st.result_layer].fw_col;


	unsigned int   node_x = pos / kernel_node_size;
	unsigned int   node_y = pos %  kernel_node_size;

	unsigned int   p_node_x = node_y / kernel_block_size;
	unsigned int   p_node_y = node_y %  kernel_block_size;

	int x = p_node_y / g_layer[st.result_layer].type.conv.kernel_col;
	int y = p_node_y % g_layer[st.result_layer].type.conv.kernel_col;

	st.result[pos] = 0;

	unsigned int   pre_node_pos = p_node_x * g_layer[st.result_layer].bw_node_size;
	unsigned int   node_pos = node_x * g_layer[st.result_layer].fw_node_size;

	unsigned int   j;
	for ( j = 0; j < nodes_len; j++)
	{
		unsigned int   batch_x = j / g_layer[st.result_layer].fw_node_size;
		unsigned int   batch_y = j % g_layer[st.result_layer].fw_node_size;

		int u = batch_y / g_layer[st.result_layer].fw_col;
		int v = batch_y % g_layer[st.result_layer].fw_col;

		if (u < g_layer[st.result_layer].type.conv.top_padding || v < g_layer[st.result_layer].type.conv.left_padding || u >= g_layer[st.result_layer].fw_row - g_layer[st.result_layer].type.conv.bottom_padding	||
			v >= g_layer[st.result_layer].fw_col - g_layer[st.result_layer].type.conv.right_padding || (u - g_layer[st.result_layer].type.conv.top_padding) * g_layer[st.result_layer].type.conv.row_stride + x >= g_layer[st.result_layer].bw_row ||
			(v - g_layer[st.result_layer].type.conv.left_padding) * g_layer[st.result_layer].type.conv.col_stride + y >= g_layer[st.result_layer].bw_col)
			continue;		
		st.result[pos] += st.in1[batch_x * g_layer[st.result_layer].bw_batch_size + pre_node_pos + ((u - g_layer[st.result_layer].type.conv.top_padding) *
			g_layer[st.result_layer].type.conv.row_stride + x ) * g_layer[st.result_layer].bw_col +((v - g_layer[st.result_layer].type.conv.left_padding) * g_layer[st.result_layer].type.conv.col_stride + y)] *
			st.in2[batch_x * g_layer[st.result_layer].fw_batch_size + node_pos + u * g_layer[st.result_layer].fw_col + v];
	}
	st.result[pos] = st.result[pos] / st.batchsize;
	return st.result[pos];
}


/*
int nodes_len = m_batch_num *m_fw_row*m_fw_col;
struct gpu_t st;
st.result = m_fw_total_input;
st.result_layer = m_gpu_layer;
st.in1 = m_bw_total_input;
st.start = 0;
st.len = nodes_len;
st.sum_result = m_kernel_diff_gpu;
st.sum_result_count= kernel_buff_size;
st.batchsize = m_batch_num;
gpu_sum(st, FN_CONV_UPDATE);
*/
/*__device__ GTYPE conv_update(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{
	unsigned int   kernel_block_size = g_layer[st.result_layer].kernel_row * g_layer[st.result_layer].kernel_col;
	unsigned int   kernel_node_size = g_layer[st.result_layer].bw_node_num * kernel_block_size;	
	
	//找到卷积核m_kernel的2维坐标
	unsigned int   node_x = pos_y / kernel_node_size;
	unsigned int   node_y = pos_y %  kernel_node_size;

	unsigned int   p_node_x = node_y / kernel_block_size;
	unsigned int   p_node_y = node_y %  kernel_block_size;

	int x = p_node_y / g_layer[st.result_layer].kernel_col;
	int y = p_node_y % g_layer[st.result_layer].kernel_col;

	//找到m_bw_total_input的2维坐标
	unsigned int   batch_x = pos_x / g_layer[st.result_layer].fw_node_size;
	unsigned int   batch_y = pos_x % g_layer[st.result_layer].fw_node_size;

	int u = batch_y / g_layer[st.result_layer].fw_col;
	int v = batch_y % g_layer[st.result_layer].fw_col;
		

	if (u < g_layer[st.result_layer].top_padding || v < g_layer[st.result_layer].left_padding || u >= g_layer[st.result_layer].fw_row - g_layer[st.result_layer].bottom_padding ||
		v >= g_layer[st.result_layer].fw_col - g_layer[st.result_layer].right_padding || (u - g_layer[st.result_layer].top_padding) * g_layer[st.result_layer].row_stride + x >= g_layer[st.result_layer].bw_row || 
		(v - g_layer[st.result_layer].left_padding) * g_layer[st.result_layer].col_stride + y >= g_layer[st.result_layer].bw_col)
		return 0;	
	else
		return  st.result[batch_x * g_layer[st.result_layer].bw_batch_size + p_node_x * g_layer[st.result_layer].bw_node_size +	
		((u - g_layer[st.result_layer].top_padding) * g_layer[st.result_layer].row_stride + x) * g_layer[st.result_layer].bw_col +	
		((v - g_layer[st.result_layer].left_padding) * g_layer[st.result_layer].col_stride + y)] *
			st.in1[batch_x * g_layer[st.result_layer].fw_batch_size + node_x * g_layer[st.result_layer].fw_node_size + u * g_layer[st.result_layer].fw_col + v] / g_layer[st.result_layer].batch_num;
}*/


/*
nodes_len = m_batch_num *m_fw_row*m_fw_col;
st.result = m_bw_total_input;
st.result_layer = m_gpu_layer;
st.start = 0;
st.len = nodes_len;
st.sum_result = m_kernel_b_diff_gpu;
st.sum_result_count = m_fw_node_num;
st.batchsize = m_batch_num;
gpu_sum(st, FN_CONV_UPDATE_SUM);
*/
__device__   GTYPE conv_update_sum(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{	
	//找到m_bw_total_input的坐标
	unsigned int   batch_x = pos_x / g_layer[st.result_layer].fw_node_size;
	unsigned int   batch_y = pos_x % g_layer[st.result_layer].fw_node_size;

	int u = batch_y / g_layer[st.result_layer].fw_col;
	int v = batch_y % g_layer[st.result_layer].fw_col;

	if (u < g_layer[st.result_layer].type.conv.top_padding || v < g_layer[st.result_layer].type.conv.left_padding ||
		u >= g_layer[st.result_layer].fw_row - g_layer[st.result_layer].type.conv.bottom_padding	|| v >= g_layer[st.result_layer].fw_col - g_layer[st.result_layer].type.conv.right_padding)
		return 0;
	return st.result[batch_x * g_layer[st.result_layer].fw_batch_size + pos_y * g_layer[st.result_layer].fw_node_size + batch_y] / st.batchsize;
}
#endif

#endif