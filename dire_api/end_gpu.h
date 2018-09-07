#pragma once
#ifndef _END_GPU__H
#define _END_GPU__H

#include "base_gpu.h"

#ifdef GPU

/*
struct gpu_t st;
st.result = m_bw_output;
st.in1 = m_fw_total_input;
st.in2 = m_label;
st.in2_layer = m_gpu_layer;
if (m_cost_type == WEIGHT_CROSS_ENTROPY)
{
st.in3 = m_label_weight_first;
st.in4 = m_label_weight_second;
}
st.start = 0;
st.len = m_bw_buff_size;
gpu_cal(st, FN_END_BW);
*/
__device__   GTYPE end_bw(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{
	unsigned  int pos = pos_y * st.len + pos_x;
	
	if (g_layer[st.in2_layer].type.end.cost_type == LSE)
	{
		st.result[pos] = st.in1[pos] - st.in2[pos];
	}
	else if (g_layer[st.in2_layer].type.end.cost_type == CROSS_ENTROPY)
	{
		st.result[pos] = (1 - st.in2[pos]) / (1 - st.in1[pos]) - st.in2[pos] / st.in1[pos]; 
		//-1.0 * (st.in2[pos] - st.in1[pos]) / (st.in1[pos] * (1 - st.in1[pos]));
	}	
	else if (g_layer[st.in2_layer].type.end.cost_type == MSE)
	{
		st.result[pos] = (st.in1[pos] - st.in2[pos]) / g_layer[st.in2_layer].fw_node_size;
	}
	else if (g_layer[st.in2_layer].type.end.cost_type == WEIGHT_CROSS_ENTROPY)
	{
		unsigned int batch_y = pos % g_layer[st.in2_layer].fw_batch_size;	
		st.result[pos] = st.in4[batch_y]* (1 - st.in2[pos]) / (1 - st.in1[pos]) - st.in3[batch_y] * st.in2[pos] / st.in1[pos];
	}	
	else
		st.result[pos] = 0;
	return st.result[pos];
}


/*
struct gpu_t st;
st.result = m_label;
st.result_layer = m_gpu_layer;
st.in1 = m_fw_total_input;
if (m_cost_type == WEIGHT_CROSS_ENTROPY)
{
st.in2 = m_label_weight_first;
st.in3 = m_label_weight_second;
}
st.per_block_result = m_temp_sum_gpu;
st.start = 0;
st.len = m_bw_buff_size;
st.sum_result = m_cost;
st.sum_result_count = 1;
st.batchsize = m_batch_num;
gpu_sum(st, FN_END_COST_SUM);
*/
__device__   GTYPE end_cost_sum( gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{
	GTYPE temp;
	unsigned  int pos = pos_y * st.len + pos_x;

	if (g_layer[st.result_layer].type.end.cost_type == LSE)
	{
		temp = (st.result[pos] - st.in1[pos])*(st.result[pos] - st.in1[pos]) / 2.0;
	}
	else if (g_layer[st.result_layer].type.end.cost_type == CROSS_ENTROPY)
	{
		temp = -1.0 * (st.result[pos] * log(st.in1[pos]) + (1 - st.result[pos])*log(1 - st.in1[pos]));
	}
	else if (g_layer[st.result_layer].type.end.cost_type == MSE)
	{
		temp = (st.result[pos] - st.in1[pos])*(st.result[pos] - st.in1[pos]) / (2.0 *g_layer[st.result_layer].fw_node_size);
	}
	else if (g_layer[st.result_layer].type.end.cost_type == WEIGHT_CROSS_ENTROPY)
	{
		unsigned int batch_y = pos % g_layer[st.result_layer].fw_batch_size;
		temp = -1.0 * (st.in2[batch_y] * st.result[pos] * log(st.in1[pos]) + st.in3[batch_y] * (1 - st.result[pos])*log(1 - st.in1[pos]));
	}	
	else
		temp = 0;
	return temp / st.batchsize;
}

#endif

#endif