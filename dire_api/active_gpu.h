#pragma once

#ifndef _ACTIVE_GPU__H
#define _ACTIVE_GPU__H

#include "base_gpu.h"


#ifdef GPU

//激励函数
__device__  GTYPE active(int type, GTYPE src, GTYPE param = 0)
{
	GTYPE answer = 0.0;
	if (type == RELU)
	{
		answer = src >= 0 ? src : 0;
	}
	else if (type == SIGMOID)
	{
		answer = 1.0 / (1.0 + exp(-1.0 *  src));
	}
	else if (type == TANH)
	{
		answer = (exp(src) - exp(-1.0*src)) / (exp(src) + exp(-1.0*src));
	}
	else if (type == SOFTMAX)
	{
		answer = exp(src) / param;
	}
	else if (type == LEAKY_RELU)
	{
		answer = src >= 0 ? src : param*src;
	}
	return answer;
}


//利用激励函数的导数计算,从而得到了后向输出。不是激励函数的导数值，是后向输出！！！
__device__ GTYPE active_derivation(int type, GTYPE in, GTYPE out, GTYPE loss_in, GTYPE param = 0)
{
	GTYPE answer = 0.0;

	if (type == RELU)
	{
		answer = in >= 0 ? loss_in : 0;
	}
	else if (type == SIGMOID)
	{
		answer = out * (1 - out) * loss_in;
	}
	else if (type == TANH)
	{
		answer = (1.0 - out * out) * loss_in;
	}
	else if (type == SOFTMAX)
	{
		answer = out*param + loss_in * out;
	}
	else if (type == LEAKY_RELU)
	{
		answer = in >= 0 ? loss_in : param * loss_in;
	}
	return answer;
}

//用于softmax的所有元素的exp之和
/*
struct gpu_t st;
st.in1 = m_fw_total_input;
st.start = 0;
st.len = m_bw_batch_size;
st.sum_result = m_softmax_total;
st.sum_result_count = m_batch_num;
gpu_sum(st, FN_ACTIVE_FW_SUM);
*/
__device__   GTYPE active_fw_sum(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{
	unsigned  int pos = pos_y * st.len + pos_x;
	return exp(st.in1[pos]);
}

//前向
/*
struct gpu_t st;
st.result = m_fw_output;
st.result_layer = m_gpu_layer;
st.in1 = m_fw_total_input;
if (m_active_type == SOFTMAX)
st.in2 = m_softmax_total;
else if (m_active_type == LEAKY_RELU)
st.in2 = m_leakey_relu_gpu;
st.start = 0;
st.len = m_fw_buff_size;
gpu_cal(st, FN_ACTIVE_FW);
*/
__device__   GTYPE active_fw(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{	

	unsigned  int pos = pos_y * st.len + pos_x;
	if (g_layer[st.result_layer].type.active.active_type == SOFTMAX)
	{
		st.result[pos] = active(g_layer[st.result_layer].type.active.active_type, st.in1[pos], st.in2[pos / g_layer[st.result_layer].fw_batch_size]);
	}
	else if (g_layer[st.result_layer].type.active.active_type == LEAKY_RELU)
	{
		st.result[pos] = active(g_layer[st.result_layer].type.active.active_type, st.in1[pos], st.in2[0]);
	}
	else
	{
		st.result[pos] = active(g_layer[st.result_layer].type.active.active_type, st.in1[pos]);
	}
	return st.result[pos];
}



//后向softmax求导时需要的累计和
/*
struct gpu_t st;
st.in1 = m_bw_total_input;
st.in2 = m_fw_output;
st.per_block_result = m_temp_sum_gpu;
st.start = 0;
st.len = m_fw_batch_size;
st.sum_result = m_softmax_total;
st.sum_result_count = m_batch_num;
gpu_sum(st, FN_ACTIVE_BW_SUM);
*/
__device__   GTYPE active_bw_sum(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{
	unsigned  int pos = pos_y * st.len + pos_x;
	return -1.0 * st.in1[pos] * st.in2[pos];
}


//后向
/*
struct gpu_t st;
st.result = m_bw_output;
st.in1 = m_fw_total_input;
st.in2 = m_fw_output;
st.in2_layer = m_gpu_layer;
st.in3 = m_bw_total_input;
if (m_active_type == SOFTMAX)
st.in4 = m_softmax_total;
else if (m_active_type == LEAKY_RELU)
st.in4 = m_leakey_relu_gpu;
st.start = 0;
st.len = m_bw_buff_size;
gpu_cal(st,FN_ACTIVE_BW);
*/
__device__   GTYPE active_bw(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{
	unsigned  int pos = pos_y * st.len + pos_x;
	
	if (g_layer[st.in2_layer].type.active.active_type == SOFTMAX)
	{
		st.result[pos] = active_derivation(g_layer[st.in2_layer].type.active.active_type, st.in1[pos], st.in2[pos], st.in3[pos], st.in4[pos / g_layer[st.in2_layer].bw_batch_size]);
	}
	else if (g_layer[st.in2_layer].type.active.active_type == LEAKY_RELU)
	{
		st.result[pos] = active_derivation(g_layer[st.in2_layer].type.active.active_type, st.in1[pos], 0, st.in3[pos], st.in4[0]);
	}
	else
	{
		if (st.in2)
			st.result[pos] = active_derivation(g_layer[st.in2_layer].type.active.active_type, st.in1[pos], st.in2[pos], st.in3[pos]);
		else
			st.result[pos] = active_derivation(g_layer[st.in2_layer].type.active.active_type, st.in1[pos], 0, st.in3[pos]);
	}
	return st.result[pos];
}


/*
struct gpu_t st;
st.result = m_bw_total_input;
st.result_layer = m_gpu_layer;
st.in1 = m_fw_total_input;
st.start = 0;
st.len = m_fw_buff_size;
st.sum_result = m_leakey_relu_diff_gpu;
st.sum_result_count = 1;
st.batchsize = m_batch_num;
gpu_sum(st, FN_ACTIVE_UPDATE_SUM);
*/
__device__   GTYPE active_update_sum(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{	
	unsigned  int pos = pos_y * st.len + pos_x;
	if ( st.in1[pos] < 0 )
		return st.result[pos] * st.in1[pos] / st.batchsize;
	return 0;
}

#endif

#endif