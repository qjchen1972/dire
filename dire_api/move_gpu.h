#pragma once
#ifndef _MOVE_GPU__H
#define _MOVE_GPU__H

#include "base_gpu.h"

#ifdef GPU
/*
struct gpu_t st;
st.result = m_fw_total_input;
st.result_layer = m_gpu_layer;
st.in1 = m_max_value;
st.in5 = m_max_pos;
st.start = 0;
st.len = m_batch_num;
gpu_cal(st, FN_MOVE_FW_MAX);
*/
__device__   GTYPE move_fw_max(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{
	unsigned  int pos = pos_y * st.len + pos_x;
	unsigned int   i;
	
	unsigned int   start = g_layer[st.result_layer].fw_batch_size * pos;
	unsigned int   end = g_layer[st.result_layer].fw_batch_size * (pos + 1);

	st.in1[pos] = st.result[start];
	st.in5[pos] = 0;

	for (i = start + 1; i < end; i++)
	{
		if (st.result[i] > st.in1[pos] )
		{
			st.in1[pos] = st.result[i];
			st.in5[pos] = i - start;
		}
	}
	return st.in1[pos];
}

/*
st.result = m_fw_output;
st.result_layer = m_gpu_layer;
st.in1 = m_fw_total_input;
st.in2 = m_max_value;
st.start = 0;
st.len = m_fw_buff_size;
gpu_cal(st, FN_MOVE_FW);
*/
__device__   GTYPE move_fw(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{
	unsigned  int pos = pos_y * st.len + pos_x;
	unsigned int   batch_x = pos / g_layer[st.result_layer].fw_batch_size;
	st.result[pos] = st.in1[pos] - st.in2[batch_x];	
	return st.result[pos];
}

/*
struct gpu_t st;
st.result = m_bw_output;
st.in1 = m_bw_total_input;
st.start = 0;
st.len = m_bw_buff_size;
gpu_cal(st, FN_MOVE_BW);
*/
__device__   GTYPE move_bw(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{
	unsigned  int pos = pos_y * st.len + pos_x;
	st.result[pos] = st.in1[pos];
	return st.result[pos];
}
#endif

#endif