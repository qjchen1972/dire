#pragma once
#ifndef _LINEARMAPPING_GPU__H
#define _LINEARMAPPING_GPU__H

#include "base_gpu.h"

#ifdef GPU

/*
struct gpu_t st;
st.result = m_fw_output;
st.result_layer = m_gpu_layer;
st.in1 = m_fw_total_input;
st.in3 = m_bda_gpu;
st.start = 0;
st.len = m_fw_buff_size;
gpu_cal(st, FN_LINEARMAPPING_FW);
*/
__device__   GTYPE linearmapping_fw(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{
	unsigned  int pos = pos_y * st.len + pos_x;
	unsigned  int  batch_x = pos / g_layer[st.result_layer].fw_batch_size;
	unsigned int   batch_y = pos % g_layer[st.result_layer].fw_batch_size;
	unsigned int bw_pos = pos % g_layer[st.result_layer].bw_batch_size;
	st.result[pos] = st.in1[batch_x * g_layer[st.result_layer].bw_batch_size + bw_pos] + st.in3[batch_y];
	return st.result[pos];
}

/*
struct gpu_t st;
st.result = m_bw_output;
st.result_layer = m_layer_num;
st.in1 = m_bw_total_input;
st.start = 0;
st.len = m_bw_buff_size;
gpu_cal(st, FN_LINEARMAPPING_BW);
*/
__device__   GTYPE linearmapping_bw(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{
	unsigned  int pos = pos_y * st.len + pos_x;
	unsigned int batch_x = pos / g_layer[st.result_layer].bw_batch_size;
	unsigned int   batch_y = pos % g_layer[st.result_layer].bw_batch_size;
	unsigned int  fw_pos = g_layer[st.result_layer].fw_batch_size / g_layer[st.result_layer].bw_batch_size;
	st.result[pos] = 0.0;
	for( int i =0; i< fw_pos; i++)
		st.result[pos] += st.in1[batch_x * g_layer[st.result_layer].fw_batch_size + i*g_layer[st.result_layer].bw_batch_size + batch_y];
	return st.result[pos];
}


/*
struct gpu_t st;
st.result = m_bda_diff_gpu;
st.in2 = m_bw_total_input;
st.in2_layer = m_layer_num;
st.in3 = m_fw_total_input;
st.start = 0;
st.len = m_fw_batch_size;
st.batchsize = m_batch_num;
gpu_cal(st, FN_LINEARMAPPING_UPDATE);
*/
__device__   GTYPE linearmapping_update(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{
	unsigned  int pos = pos_y * st.len + pos_x;	
	unsigned int i;

	st.result[pos] = 0;
	for (i = 0; i < st.batchsize; i++)
	{
		unsigned int   p = i * g_layer[st.in2_layer].fw_batch_size + pos;
		st.result[pos] += st.in2[p];
	}
	st.result[pos] = st.result[pos] / st.batchsize;
	return st.result[pos];
}

#endif
#endif