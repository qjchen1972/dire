#pragma once
#ifndef _GRATING_GPU__H
#define _GRATING_GPU__H

#include "base_gpu.h"

#ifdef GPU

/*
struct gpu_t st;
st.result = m_fw_output;
st.result_layer = m_gpu_layer;
st.in1 = m_fw_total_input;
st.start = 0;
st.len = m_fw_buff_size;
gpu_cal(st, FN_GRATING_FW);
*/
__device__   GTYPE grating_fw(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{	
	unsigned  int pos = pos_y * st.len + pos_x;
	if(g_layer[st.result_layer].type.grating.grating_type == GRATING_TYPE )
		st.result[pos] = st.in1[pos];
	else
	{		
		int move_x = (g_layer[st.result_layer].bw_row - g_layer[st.result_layer].fw_row) / 2;
		int move_y = (g_layer[st.result_layer].bw_col - g_layer[st.result_layer].fw_col) / 2;
		unsigned int   batch_x = pos / g_layer[st.result_layer].fw_batch_size;
		unsigned int   batch_y = pos % g_layer[st.result_layer].fw_batch_size;
		unsigned int   node_x = batch_y / g_layer[st.result_layer].fw_node_size;
		unsigned int   node_y = batch_y % g_layer[st.result_layer].fw_node_size;
		int x = node_y / g_layer[st.result_layer].fw_col;
		int y = node_y % g_layer[st.result_layer].fw_col;
		st.result[pos] = st.in1[batch_x * g_layer[st.result_layer].bw_batch_size + node_x * g_layer[st.result_layer].bw_node_size +	
			(x + move_x) * g_layer[st.result_layer].bw_col + y + move_y];
	}	
	return st.result[pos];
}

/*
struct gpu_t st;
st.result = m_bw_output;
st.result_layer = m_gpu_layer;
st.in1 = m_bw_total_input;
st.start = 0;
st.len = m_bw_buff_size;
gpu_cal(st, FN_GRATING_BW);
*/
__device__   GTYPE grating_bw(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{	
	unsigned  int pos = pos_y * st.len + pos_x;
	if( g_layer[st.result_layer].type.grating.grating_type == GRATING_TYPE)
		st.result[pos] = st.in1[pos];
	else
	{		
		int move_x = (g_layer[st.result_layer].bw_row - g_layer[st.result_layer].fw_row) / 2;
		int move_y = (g_layer[st.result_layer].bw_col - g_layer[st.result_layer].fw_col) / 2;
		unsigned int   batch_x = pos / g_layer[st.result_layer].bw_batch_size;
		unsigned int   batch_y = pos % g_layer[st.result_layer].bw_batch_size;
		unsigned int   node_x = batch_y / g_layer[st.result_layer].bw_node_size;
		unsigned int   node_y = batch_y % g_layer[st.result_layer].bw_node_size;
		int x = node_y / g_layer[st.result_layer].bw_col;
		int y = node_y % g_layer[st.result_layer].bw_col;

		if (x < move_x || x >= move_x + g_layer[st.result_layer].fw_row || y < move_y || y >= move_y + g_layer[st.result_layer].fw_col)
			st.result[pos] = 0;
		else
			st.result[pos] = st.in1[batch_x * g_layer[st.result_layer].fw_batch_size + node_x * g_layer[st.result_layer].fw_node_size +	
			(x - move_x) * g_layer[st.result_layer].fw_col + y - move_y];
	}
	return st.result[pos];
}
#endif
#endif