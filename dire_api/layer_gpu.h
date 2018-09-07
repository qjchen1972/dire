#pragma once
#ifndef _LAYER_GPU__H
#define _LAYER_GPU__H

#include "base_gpu.h"

#ifdef GPU

/*
struct gpu_t st;
st.result = m_fw_total_input;
st.result_layer = m_prev_layer[0]->m_layer_num;

unsigned int in[MAX_PREV_NUM];
memset(in, 0, sizeof(unsigned int)*MAX_PREV_NUM);
unsigned int in_layer[MAX_PREV_NUM];

for (size = 0; size < m_prev_num; size++)
{
in[size] = m_prev_layer[size]->m_fw_output;
in_layer[size] = m_prev_layer[size]->m_layer_num;
}
HANDLE_ERROR(cudaMemcpy(m_in_buf, in, sizeof(unsigned int) *MAX_PREV_NUM, cudaMemcpyHostToDevice));
HANDLE_ERROR(cudaMemcpy(m_in_layer_buf, in_layer, sizeof(unsigned int) *MAX_PREV_NUM, cudaMemcpyHostToDevice));
st.in = m_in_buf;
st.in_layer = m_in_layer_buf;

st.start = 0;
st.len = m_prev_layer[0]->m_fw_buff_size;
gpu_cal(st, FN_RESNET_LAYER_ADD);
*/
__device__   GTYPE resnet_layer_add(gpu_t st, unsigned int pos_x, unsigned int pos_y)
{
	for (int i = 0; i < MAX_CONN_NUM; i++)
	{
		if (!st.in[i]) break;
		if (g_layer[st.result_layer].fw_batch_size == g_layer[st.in_layer[i]].fw_batch_size)
		{
			if(i == 0)
				st.result[pos_x] = ((GTYPE*)st.in[i])[pos_x];
			else
				st.result[pos_x] += ((GTYPE*)st.in[i])[pos_x];
		}
		else if (g_layer[st.result_layer].fw_batch_size > g_layer[st.in_layer[i]].fw_batch_size)
		{
			unsigned int batch_x = pos_x / g_layer[st.result_layer].fw_batch_size;
			unsigned int batch_y = pos_x % g_layer[st.result_layer].fw_batch_size;
			unsigned int node_x = batch_y / g_layer[st.result_layer].fw_node_size;
			unsigned int node_y = batch_y % g_layer[st.result_layer].fw_node_size;

			int x = node_y / g_layer[st.result_layer].fw_col;
			int y = node_y % g_layer[st.result_layer].fw_col;
			if (i == 0)
			{
				if (x % 2 == 0 && y % 2 == 0 && x / 2 < g_layer[st.in_layer[i]].fw_row && y / 2 < g_layer[st.in_layer[i]].fw_col)
					st.result[pos_x] = ((GTYPE*)st.in[i])[batch_x * g_layer[st.in_layer[i]].fw_batch_size + node_x* g_layer[st.in_layer[i]].fw_node_size +
						x* g_layer[st.in_layer[i]].fw_col / 2 + y / 2];
				else
						st.result[pos_x] = 0;
			}
			else
			{
				if (x % 2 == 0 && y % 2 == 0 && x / 2 < g_layer[st.in_layer[i]].fw_row && y / 2 < g_layer[st.in_layer[i]].fw_col)
					st.result[pos_x] += ((GTYPE*)st.in[i])[batch_x * g_layer[st.in_layer[i]].fw_batch_size + node_x* g_layer[st.in_layer[i]].fw_node_size +
					x* g_layer[st.in_layer[i]].fw_col / 2 + y / 2];
			}		
		}
		else
		{
			unsigned int batch_x = pos_x / g_layer[st.result_layer].fw_batch_size;
			unsigned int batch_y = pos_x % g_layer[st.result_layer].fw_batch_size;
			unsigned int node_x = batch_y / g_layer[st.result_layer].fw_node_size;
			unsigned int node_y = batch_y % g_layer[st.result_layer].fw_node_size;

			int x = node_y / g_layer[st.result_layer].fw_col;
			int y = node_y % g_layer[st.result_layer].fw_col;

			if (2 * x < g_layer[st.in_layer[i]].fw_row && 2 * y < g_layer[st.in_layer[i]].fw_col)
				if(i == 0)
					st.result[pos_x] = ((GTYPE*)st.in[i])[batch_x* g_layer[st.in_layer[i]].fw_batch_size + node_x*g_layer[st.in_layer[i]].fw_node_size +
					(2 * x)*g_layer[st.in_layer[i]].fw_col + 2 * y];
				else
					st.result[pos_x] += ((GTYPE*)st.in[i])[batch_x* g_layer[st.in_layer[i]].fw_batch_size + node_x*g_layer[st.in_layer[i]].fw_node_size +
					(2 * x)*g_layer[st.in_layer[i]].fw_col + 2 * y];			
		}

	}
	return st.result[pos_x];
}

/*
struct gpu_t st;
st.result = m_fw_total_input;
st.result_layer = m_layer_num;

unsigned int in[MAX_PREV_NUM];
memset(in, 0, sizeof(unsigned int)*MAX_PREV_NUM);
unsigned int in_layer[MAX_PREV_NUM];

for (size = 0; size < m_prev_num; size++)
{
in[size] = m_prev_layer[size]->m_fw_output;
in_layer[size] = m_prev_layer[size]->m_layer_num;
}
HANDLE_ERROR(cudaMemcpy(m_in_buf, in, sizeof(unsigned int) *MAX_PREV_NUM, cudaMemcpyHostToDevice));
HANDLE_ERROR(cudaMemcpy(m_in_layer_buf, in_layer, sizeof(unsigned int) *MAX_PREV_NUM, cudaMemcpyHostToDevice));
st.in = m_in_buf;
st.in_layer = m_in_layer_buf;
st.start = 0;
st.len = m_bw_buff_size;
gpu_cal(st, FN_DENSENET_LAYER_FW_INPUT);
*/
__device__   GTYPE densenet_layer_fw_input(gpu_t st, unsigned int pos_x, unsigned int pos_y)
{
	unsigned int batch_x = pos_x / g_layer[st.result_layer].bw_batch_size;
	unsigned int batch_y = pos_x % g_layer[st.result_layer].bw_batch_size;

	unsigned int len = 0;
	st.result[pos_x] = 0;
	for (int i = 0; i < MAX_CONN_NUM; i++)
	{
		if (!st.in[i]) break;
		if (batch_y < len + g_layer[st.in_layer[i]].fw_batch_size)
		{
			st.result[pos_x] = ((GTYPE*)st.in[i])[batch_x*g_layer[st.in_layer[i]].fw_batch_size + batch_y - len];
			break;
		}
		len += g_layer[st.in_layer[i]].fw_batch_size;
	}
	return st.result[pos_x];
}

/*
struct gpu_t st;
st.result = m_bw_total_input;
st.result_layer = m_layer_num;

unsigned int in[MAX_PREV_NUM];
memset(in, 0, sizeof(unsigned int)*MAX_PREV_NUM);
unsigned int in_layer[MAX_PREV_NUM];
unsigned int in_pos[MAX_PREV_NUM];

for (size = 0; size < m_next_num; size++)
{
in[size] = m_next_layer[size]->m_bw_output;
in_layer[size] = m_next_layer[size]->m_layer_num;
in_pos[size] = 0;
for (int n = 0; n < m_next_pos[size]; n++)
in_pos[size] += m_next_layer[size]->m_prev_layer[n]->m_fw_batch_size;
}
HANDLE_ERROR(cudaMemcpy(m_in_buf, in, sizeof(unsigned int) *MAX_PREV_NUM, cudaMemcpyHostToDevice));
HANDLE_ERROR(cudaMemcpy(m_in_layer_buf, in_layer, sizeof(unsigned int) *MAX_PREV_NUM, cudaMemcpyHostToDevice));
HANDLE_ERROR(cudaMemcpy(m_in_pos_buf, in_pos, sizeof(unsigned int) *MAX_PREV_NUM, cudaMemcpyHostToDevice));
st.in = m_in_buf;
st.in_layer = m_in_layer_buf;
st.in_pos = m_in_pos_buf;
st.start = 0;
st.len = m_fw_buff_size;
gpu_cal(st, FN_DENSENET_LAYER_BW_INPUT);
*/
__device__   GTYPE densenet_layer_bw_input(gpu_t st, unsigned int pos_x, unsigned int pos_y)
{
	unsigned int batch_x = pos_x / g_layer[st.result_layer].fw_batch_size;
	unsigned int batch_y = pos_x % g_layer[st.result_layer].fw_batch_size;

	for (int i = 0; i < MAX_CONN_NUM; i++)
	{
		if (!st.in[i]) break;
		if (i == 0)
			st.result[pos_x] = ((GTYPE*)st.in[i])[batch_x*g_layer[st.in_layer[i]].bw_batch_size + st.in_pos[i] + batch_y];
		else
			st.result[pos_x] += ((GTYPE*)st.in[i])[batch_x*g_layer[st.in_layer[i]].bw_batch_size + st.in_pos[i] + batch_y];
	}
	return st.result[pos_x];
}

#endif
#endif