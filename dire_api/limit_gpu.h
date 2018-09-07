#pragma once
#ifndef _LIMIT_GPU__H
#define _LIMIT_GPU__H

#include "base_gpu.h"

#ifdef GPU
/*
struct gpu_t st;
st.result = m_fw_output;
st.in1 = m_fw_total_input;
st.start = 0;
st.len = m_fw_buff_size;
gpu_cal(st, FN_LIMIT_FW);
*/
__device__   GTYPE limit_fw(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{	
	unsigned  int pos = pos_y * st.len + pos_x;
	if ( st.in1[pos] > LIMIT_MAX)
		st.result[pos] = LIMIT_MAX;
	else if (st.in1[pos] < LIMIT_MIN)
		st.result[pos] = LIMIT_MIN;
	else
		st.result[pos] = st.in1[pos];
	return st.result[pos];
}

/*
struct gpu_t st;
st.result = m_bw_output;
st.in1 = m_fw_total_input;
st.in2 = m_bw_total_input;
st.start = 0;
st.len = m_bw_buff_size;
gpu_cal(st, FN_LIMIT_BW);
*/
__device__   GTYPE limit_bw(gpu_t st, unsigned int   pos_x, unsigned int pos_y)
{	
	unsigned  int pos = pos_y * st.len + pos_x;
	if ( st.in1[pos] > LIMIT_MAX)
		st.result[pos] = 0;
	else if ( st.in1[pos] < LIMIT_MIN)
		st.result[pos] = 0;
	else
		st.result[pos] = st.in2[pos];
	return st.result[pos];
}
#endif

#endif