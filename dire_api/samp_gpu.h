#pragma once
#ifndef _SAMP_GPU__H
#define _SAMP_GPU__H

#include "base_gpu.h"

#ifdef GPU

 __device__  GTYPE sample(GTYPE *fw_total_input, unsigned int   *max_pos, int pool_type, int pool_row, int pool_col,
	int prev_row, int prev_col, unsigned int  src_pos, int src_x, int src_y)
{
	 GTYPE temp = 0;
	int m, n;

	if (pool_type == MAX_POOL)
	{
		temp = fw_total_input[src_pos + src_x * prev_col + src_y];
		max_pos[src_pos + src_x * prev_col + src_y] = src_pos + src_x * prev_col + src_y;
		for (m = 0; m < pool_row; m++)
			for (n = 0; n < pool_col; n++)
			{
				if (src_x + m < prev_row && src_y + n < prev_col)
				{
					if (fw_total_input[src_pos + (src_x + m) * prev_col + src_y + n] > temp)
					{
						temp = fw_total_input[src_pos + (src_x + m) * prev_col + src_y + n];
						max_pos[src_pos + src_x * prev_col + src_y] = src_pos + (src_x + m) * prev_col + src_y + n;
					}
				}
			}	
	}
	else if (pool_type == AVG_POOL || pool_type == GLOBAL_AVG_POOL)
	{
		for (m = 0; m < pool_row; m++)
			for (n = 0; n < pool_col; n++)
			{
				if (src_x + m < prev_row && src_y + n < prev_col)
				{
					temp += fw_total_input[src_pos + (src_x + m) * prev_col + src_y + n];
				}
			}
		temp = temp / (pool_row * pool_col);
	}
	else
		temp = 0;

	return temp;
}

__device__ GTYPE get_bw_pool_matrix_one(GTYPE* bw_total_input, unsigned int   *max_pos, int pool_type, int pool_row, int pool_col,
	int row_stride, int col_stride, int top_padding, int left_padding, int fw_real_row, int fw_real_col, int fw_row, int fw_col,
	int prev_row, int prev_col, unsigned int   src_pos, int u, int v, unsigned int   loss_pos)
{
	GTYPE answer = 0;
	int m, n;

	for (m = 0; m < pool_row; m++)
		for (n = 0; n < pool_col; n++)
		{
			if (u - m >= 0 && v - n >= 0 && (u - m) / row_stride < fw_real_row && (v - n) / col_stride < fw_real_col &&
				(u - m) % row_stride == 0 && (v - n) % col_stride == 0)
			{
				if (pool_type == MAX_POOL)
				{
					int src_x = u - m;
					int src_y = v - n;
					GTYPE drev = 0;
					if (max_pos[src_pos + src_x*prev_col + src_y] == src_pos + u*prev_col + v) drev = 1;
					answer += bw_total_input[loss_pos + ((u - m) / row_stride + top_padding)*fw_col + (v - n) / col_stride + left_padding] * drev;
				}
				else if (pool_type == AVG_POOL || pool_type == GLOBAL_AVG_POOL)
				{
					GTYPE drev = 1.0 / (pool_row*pool_col);
					answer += bw_total_input[loss_pos + ((u - m) / row_stride + top_padding)*fw_col + (v - n) / col_stride + left_padding] * drev;
				}
				else
					answer = 0;
			}
		}
	return answer;
}


/*
struct gpu_t st;
st.result = m_fw_output;
st.result_layer = m_gpu_layer;
st.in1 = m_fw_total_input;
st.in2 = m_weight_gpu;
if (m_biasinuse)	st.in3 = m_b_gpu;
if(m_pool_type == MAX_POOL)
st.in5 = m_max_pos;
st.start = 0;
st.len = m_fw_buff_size;
gpu_cal(st, FN_SAMP_FW);
*/
__device__   GTYPE samp_fw(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{
	unsigned  int pos = pos_y * st.len + pos_x;

	unsigned int   batch_x = pos / g_layer[st.result_layer].fw_batch_size;
	unsigned int   batch_y = pos % g_layer[st.result_layer].fw_batch_size;
	unsigned int   node_x = batch_y / g_layer[st.result_layer].fw_node_size;
	unsigned int   node_y = batch_y % g_layer[st.result_layer].fw_node_size;
	int u = node_y / g_layer[st.result_layer].fw_col;
	int v = node_y % g_layer[st.result_layer].fw_col;
	
	if (u < g_layer[st.result_layer].type.samp.top_padding || u >= g_layer[st.result_layer].fw_row - g_layer[st.result_layer].type.samp.bottom_padding	||
		v < g_layer[st.result_layer].type.samp.left_padding || v >= g_layer[st.result_layer].fw_col - g_layer[st.result_layer].type.samp.right_padding )
	{
		st.result[pos] = 0;
	}
	else
	{
		unsigned int    src_pos = batch_x* g_layer[st.result_layer].bw_batch_size + node_x * g_layer[st.result_layer].bw_node_size;
		int src_x = (u - g_layer[st.result_layer].type.samp.top_padding)* g_layer[st.result_layer].type.samp.row_stride;
		int src_y = (v - g_layer[st.result_layer].type.samp.left_padding)* g_layer[st.result_layer].type.samp.col_stride;
		st.result[pos] = sample(st.in1, st.in5, g_layer[st.result_layer].type.samp.pool_type, g_layer[st.result_layer].type.samp.pool_row, g_layer[st.result_layer].type.samp.pool_col,
			g_layer[st.result_layer].bw_row, g_layer[st.result_layer].bw_col, src_pos, src_x, src_y) * st.in2[node_x];
		if(st.in3 ) st.result[pos] += st.in3[node_x];
	}
	return st.result[pos];
}


/*
struct gpu_t st;
st.result = m_bw_output;
st.result_layer = m_gpu_layer;
st.in1 = m_bw_total_input;
st.in2 = m_weight_gpu;
if (m_pool_type == MAX_POOL)
st.in5 = m_max_pos;
st.start = 0;
st.len = m_bw_buff_size;
gpu_cal(st, FN_SAMP_BW);
*/
__device__  GTYPE samp_bw(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{	
	unsigned  int pos = pos_y * st.len + pos_x;

	unsigned int   batch_x = pos / g_layer[st.result_layer].bw_batch_size;
	unsigned int   batch_y = pos % g_layer[st.result_layer].bw_batch_size;
	unsigned int   node_x = batch_y / g_layer[st.result_layer].bw_node_size;
	unsigned int   node_y = batch_y % g_layer[st.result_layer].bw_node_size;

	int u = node_y / g_layer[st.result_layer].bw_col;
	int v = node_y % g_layer[st.result_layer].bw_col;

	unsigned int   src_pos = batch_x * g_layer[st.result_layer].bw_batch_size + node_x * g_layer[st.result_layer].bw_node_size;
	unsigned int    loss_pos = batch_x* g_layer[st.result_layer].fw_batch_size + node_x * g_layer[st.result_layer].fw_node_size;

	st.result[pos] = get_bw_pool_matrix_one(st.in1, st.in5, g_layer[st.result_layer].type.samp.pool_type, g_layer[st.result_layer].type.samp.pool_row,
		g_layer[st.result_layer].type.samp.pool_col, g_layer[st.result_layer].type.samp.row_stride, g_layer[st.result_layer].type.samp.col_stride, g_layer[st.result_layer].type.samp.top_padding,
		g_layer[st.result_layer].type.samp.left_padding, g_layer[st.result_layer].fw_real_row, g_layer[st.result_layer].fw_real_col,
		g_layer[st.result_layer].fw_row, g_layer[st.result_layer].fw_col, g_layer[st.result_layer].bw_row, g_layer[st.result_layer].bw_col,
		src_pos, u, v, loss_pos) * st.in2[node_x];
	return st.result[pos];
}

/*
nodes_len = m_batch_num * m_fw_row * m_fw_col
struct gpu_t st;
st.result = m_bw_total_input;
st.result_layer = m_gpu_layer;
st.start = 0;
st.len = nodes_len;
st.sum_result = m_b_diff_gpu;
st.sum_result_count = m_fw_node_num;
st.batchsize = m_batch_num;
gpu_sum(st, FN_SAMP_UPDATE_B_SUM);
*/
__device__   GTYPE samp_update_b_sum(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{	
	unsigned int   batch_x = pos_x / g_layer[st.result_layer].fw_node_size;
	unsigned int   batch_y = pos_x % g_layer[st.result_layer].fw_node_size;

	int x = batch_y / g_layer[st.result_layer].fw_col;
	int y = batch_y % g_layer[st.result_layer].fw_col;

	if (x < g_layer[st.result_layer].type.samp.top_padding || x >= g_layer[st.result_layer].fw_row - g_layer[st.result_layer].type.samp.bottom_padding ||
		y < g_layer[st.result_layer].type.samp.left_padding || y >= g_layer[st.result_layer].fw_col - g_layer[st.result_layer].type.samp.right_padding)
		return 0;
	return st.result[batch_x * g_layer[st.result_layer].fw_batch_size + pos_y * g_layer[st.result_layer].fw_node_size + batch_y] / st.batchsize;
}

/*
st.result = m_bw_total_input;
st.result_layer = m_gpu_layer;
st.in1 = m_fw_total_input;
if (m_pool_type == MAX_POOL)
st.in5 = m_max_pos;
st.start = 0;
st.len = nodes_len;
st.sum_result = m_weight_diff_gpu;
st.sum_result_count = m_fw_node_num;
st.batchsize = m_batch_num;
gpu_sum(st, FN_SAMP_UPDATE_WEIGHT_SUM);
*/
__device__   GTYPE samp_update_weight_sum(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{
	unsigned int   batch_x = pos_x / g_layer[st.result_layer].fw_node_size;
	unsigned int   batch_y = pos_x % g_layer[st.result_layer].fw_node_size;

	int x = batch_y / g_layer[st.result_layer].fw_col;
	int y = batch_y % g_layer[st.result_layer].fw_col;

	if (x < g_layer[st.result_layer].type.samp.top_padding || x >= g_layer[st.result_layer].fw_row - g_layer[st.result_layer].type.samp.bottom_padding ||
		y < g_layer[st.result_layer].type.samp.left_padding || y >= g_layer[st.result_layer].fw_col - g_layer[st.result_layer].type.samp.right_padding)
		return 0;
	unsigned int    src_pos = batch_x * g_layer[st.result_layer].bw_batch_size + pos_y * g_layer[st.result_layer].bw_node_size;
	int src_x = (x - g_layer[st.result_layer].type.samp.top_padding) * g_layer[st.result_layer].type.samp.row_stride;
	int src_y = (y - g_layer[st.result_layer].type.samp.left_padding) * g_layer[st.result_layer].type.samp.col_stride;

	return st.result[batch_x * g_layer[st.result_layer].fw_batch_size + pos_y * g_layer[st.result_layer].fw_node_size + batch_y] *
		sample(st.in1, st.in5, g_layer[st.result_layer].type.samp.pool_type, g_layer[st.result_layer].type.samp.pool_row, g_layer[st.result_layer].type.samp.pool_col,
			g_layer[st.result_layer].bw_row, g_layer[st.result_layer].bw_col, src_pos, src_x, src_y) / st.batchsize;
}

#endif
#endif