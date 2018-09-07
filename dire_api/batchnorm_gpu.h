#pragma once
#ifndef _BATCHNORM_GPU__H
#define _BATCHNORM_GPU__H

#include "base_gpu.h"

#ifdef GPU

/*
elv_num = m_batch_num * m_bw_node_size;
st.result = m_fw_total_input;
st.result_layer = m_gpu_layer;
st.start = 0;
st.len = elv_num;
st.sum_result = m_ex_gpu;
st.sum_result_count = m_bw_node_num;
gpu_sum(st, FN_BN_FW_EX_SUM);
*/
__device__   GTYPE bn_fw_ex_sum(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{	
	unsigned int   batch_x = pos_x / g_layer[st.result_layer].bw_node_size;
	unsigned int   batch_y = pos_x % g_layer[st.result_layer].bw_node_size;

	return st.result[batch_x *  g_layer[st.result_layer].bw_batch_size + pos_y*g_layer[st.result_layer].bw_node_size + batch_y] / st.len;
}

/*
elv_num = m_batch_num * m_bw_node_size;
st.result = m_fw_total_input;
st.result_layer = m_gpu_layer;
st.in1 = m_ex_gpu;
st.start = 0;
st.len = elv_num;
st.sum_result = m_var_gpu;
st.sum_result_count = m_bw_node_num;
gpu_sum(st, FN_BN_FW_VAR_SUM);
*/
__device__   GTYPE bn_fw_var_sum(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{	
	unsigned int   batch_x = pos_x / g_layer[st.result_layer].bw_node_size;
	unsigned int   batch_y = pos_x % g_layer[st.result_layer].bw_node_size;
	return (st.result[batch_x * g_layer[st.result_layer].bw_batch_size + pos_y*g_layer[st.result_layer].bw_node_size + batch_y] - st.in1[pos_y])*
		(st.result[batch_x * g_layer[st.result_layer].bw_batch_size + pos_y*g_layer[st.result_layer].bw_node_size + batch_y] - st.in1[pos_y]) / st.len;
}

/*
st.result = m_fw_output;
st.result_layer = m_gpu_layer;
st.in1 = m_fw_total_input;
st.in2 = m_ex_gpu;
st.in3 = m_var_gpu;
st.start = 0;
st.len = m_fw_buff_size;
gpu_cal(st, FN_BN_FW);
*/
__device__  GTYPE bn_fw(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{	
	unsigned  int pos = pos_y * st.len + pos_x;
	unsigned int   batch_y = pos % g_layer[st.result_layer].fw_batch_size;
	unsigned int   node_x = batch_y / g_layer[st.result_layer].fw_node_size;
	st.result[pos] = (st.in1[pos] - st.in2[node_x]) / sqrt(st.in3[node_x] + INFINIT_NUM);
	return st.result[pos];
}

/*
elv_num = m_batch_num * m_fw_node_size
struct gpu_t st;
st.result = m_bw_total_input;
st.result_layer = m_gpu_layer;
st.start = 0;
st.len = elv_num;
st.sum_result = m_bw_ex;
st.sum_result_count = m_fw_node_num;
gpu_sum(st, FN_BN_BW_EX_SUM);
*/
__device__   GTYPE bn_bw_ex_sum(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{	
	unsigned int   batch_x = pos_x / g_layer[st.result_layer].fw_node_size;
	unsigned int   batch_y = pos_x % g_layer[st.result_layer].fw_node_size;
	return st.result[batch_x *  g_layer[st.result_layer].fw_batch_size + pos_y*g_layer[st.result_layer].fw_node_size + batch_y] / st.len;
}

/*
elv_num = m_batch_num * m_fw_node_size
st.result = m_bw_total_input;
st.result_layer = m_gpu_layer;
st.in1 = m_fw_output;
st.start = 0;
st.len = elv_num;
st.sum_result = m_bw_multex;
st.sum_result_count = m_fw_node_num;
gpu_sum(st, FN_BN_BW_EXMULTI_SUM);
*/
__device__   GTYPE bn_bw_exmulti_sum(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{	
	unsigned int   batch_x = pos_x / g_layer[st.result_layer].fw_node_size;
	unsigned int   batch_y = pos_x % g_layer[st.result_layer].fw_node_size;

	return st.result[batch_x *  g_layer[st.result_layer].fw_batch_size + pos_y*g_layer[st.result_layer].fw_node_size + batch_y] * 
		st.in1[batch_x *  g_layer[st.result_layer].fw_batch_size + pos_y*g_layer[st.result_layer].fw_node_size + batch_y] / st.len;
}

/*
st.result = m_bw_output;
st.result_layer = m_gpu_layer;
st.in1 = m_bw_total_input;
st.in2 = m_fw_output;
st.in3 = m_bw_ex;
st.in4 = m_bw_multex;
st.sum_result = m_var_gpu;
st.start = 0;
st.len = m_bw_buff_size;
gpu_cal(st, FN_BN_BW);
*/
__device__   GTYPE bn_bw(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{
	unsigned  int pos = pos_y * st.len + pos_x;
	unsigned int   batch_y = pos % g_layer[st.result_layer].bw_batch_size;
	unsigned int   node_x = batch_y / g_layer[st.result_layer].bw_node_size;

	st.result[pos] = (st.in1[pos] - st.in3[node_x] - st.in4[node_x] * st.in2[pos]) / sqrt(st.sum_result[node_x] + INFINIT_NUM);
	return st.result[pos];
}
#endif
#endif