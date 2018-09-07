#pragma once
#ifndef _SCALE_GPU__H
#define _SCALE_GPU__H

#include "base_gpu.h"

#ifdef GPU

/*
struct gpu_t st;
st.result = m_fw_output;
st.result_layer = m_gpu_layer;
st.in1 = m_fw_total_input;
st.in2 = m_gem_gpu;
st.in3 = m_bda_gpu;
st.start = 0;
st.len = m_fw_buff_size;
gpu_cal(st, FN_SCALE_FW);
*/
__device__   GTYPE scale_fw(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{	
	unsigned  int pos = pos_y * st.len + pos_x;
	unsigned int   batch_y = pos % g_layer[st.result_layer].fw_batch_size;
	unsigned int   node_x = batch_y / g_layer[st.result_layer].fw_node_size;
	st.result[pos] = st.in2[node_x] * st.in1[pos] + st.in3[node_x];
	return st.result[pos];
}

/*
struct gpu_t st;
st.result = m_bw_output;
st.result_layer = m_gpu_layer;
st.in1 = m_bw_total_input;
st.in2 = m_gem_gpu;
st.start = 0;
st.len = m_bw_buff_size;
gpu_cal(st, FN_SCALE_BW);
*/
__device__   GTYPE scale_bw(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{
	unsigned  int pos = pos_y * st.len + pos_x;
	unsigned int   batch_y = pos % g_layer[st.result_layer].bw_batch_size;
	unsigned int   node_x = batch_y / g_layer[st.result_layer].bw_node_size;
	st.result[pos] = st.in2[node_x] * st.in1[pos];
	return st.result[pos];
}


/*
struct gpu_t st;
st.result = m_bw_total_input;
st.result_layer = m_gpu_layer;
st.start = 0;
st.len = nodes_len;
st.sum_result = m_bda_diff_gpu;
st.sum_result_count = m_fw_node_num;
st.batchsize = m_batch_num;
gpu_sum(st, FN_SCALE_UPDATE_BDA_SUM);
*/
__device__   GTYPE scale_update_bda_sum(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{
	unsigned int   batch_x = pos_x / g_layer[st.result_layer].fw_node_size;
	unsigned int   batch_y = pos_x % g_layer[st.result_layer].fw_node_size;
	return st.result[batch_x *g_layer[st.result_layer].fw_batch_size + pos_y * g_layer[st.result_layer].fw_node_size + batch_y] / st.batchsize;
}

/*
struct gpu_t st;
st.result = m_bw_total_input;
st.result_layer = m_gpu_layer;
st.in1 = m_fw_total_input;
st.start = 0;
st.len = nodes_len;
st.sum_result = m_gem_diff_gpu;
st.sum_result_count = m_fw_node_num;
st.batchsize = m_batch_num;
gpu_sum(st, FN_SCALE_UPDATE_GEM_SUM);
*/
__device__   GTYPE scale_update_gem_sum(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{	
	unsigned int   batch_x = pos_x / g_layer[st.result_layer].fw_node_size;
	unsigned int   batch_y = pos_x % g_layer[st.result_layer].fw_node_size;
	unsigned int   p = batch_x * g_layer[st.result_layer].fw_batch_size + pos_y  * g_layer[st.result_layer].fw_node_size + batch_y;
	return st.result[p] * st.in1[p] / st.batchsize;
}

#endif
#endif