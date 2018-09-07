/*
* Copyright (c) 2017
* All rights reserved.
*
* 文件名称: move.h
* 文件标识:
* 摘    要: 线性的平移数据。减去数据中的最大值。用于softmax之前，避免softmax和cross-entropy溢出
*            这个比缩小比率的方式好，可以保证误差传递不被缩小.只用于softmax之前。切记！！！
*
* 当前版本: 1.0
* 作    者: chen.qian.jiang
* 开始日期: 2017-09-20
* 完成日期:
* 其它说明：
* 修改日期        版本号     修改人          修改内容
* -----------------------------------------------
* 2017/09/20      V1.0       陈黔江          创建
*/

#pragma once
#ifndef _MOVE__H
#define _MOVE__H

#include "layer.h"

class Move : public Clayer
{
public:
	Move()
	{
		m_layer_type = MOVE_LAYER;
	}

	~Move() {}

	void init_space()
	{
#ifdef GPU
		set_const();
		HANDLE_ERROR(cudaMalloc((void **)&m_max_value, sizeof(GTYPE) *m_batch_num));
		HANDLE_ERROR(cudaMalloc((void **)&m_max_pos, sizeof(unsigned int  )*m_batch_num));
		HANDLE_ERROR(cudaMalloc((void **)&m_max_bw, sizeof(GTYPE) *m_batch_num));
#else
		m_max_value = new GTYPE[m_batch_num];
		m_max_pos = new unsigned int  [m_batch_num];
		m_max_bw = new GTYPE[m_batch_num];
#endif
	}

	void init()
	{
		init_size();
	}


	void forward_proc()
	{
		if (m_prev_num <= 0) return;
		create_fw_input();
		get_fw_input();
		create_fw_output();

#ifdef GPU
		struct gpu_t st;
		st.result = m_fw_total_input;
		st.result_layer = m_layer_num;
		st.in1 = m_max_value;
		st.in5 = m_max_pos;
		st.start = 0;
		st.len = m_batch_num;
		gpu_cal(st, FN_MOVE_FW_MAX);

		st.result = m_fw_output;
		st.result_layer = m_layer_num;
		st.in1 = m_fw_total_input;
		st.in2 = m_max_value;
		st.start = 0;
		st.len = m_fw_buff_size;
		gpu_cal(st, FN_MOVE_FW);

#else
		unsigned int   i;
		for (i = 0; i < m_batch_num; i++)
		{
			get_max(m_fw_total_input + i*m_fw_batch_size, i, m_fw_batch_size);
		}

		for (i = 0; i < m_fw_buff_size; i++)
		{
			unsigned int   batch_x = i / m_fw_batch_size;
			m_fw_output[i] = m_fw_total_input[i] - m_max_value[batch_x];
		}
#endif
		if (g_config.m_free_type != NERVER_FREE)
		{
			free_fw_input();
			free_prev_space();
		}
	}

	void backword_proc()
	{
		if (m_next_num <= 0) return;
		create_bw_input();
		get_bw_input();
		create_bw_output();

		/*
		最大值位置的残差应该是其他值的残差之和的相反数。但若是move在softmax之前,直接等于softmax的同位置的残差。因为softmax的残差总和=0。
		由于move只用于softmax之前。所以实现时就直接赋值了。若是用在其他模块之前。这个残差计算不对
		*/
#ifdef GPU
		struct gpu_t st;		
		st.result = m_bw_output;
		st.in1 = m_bw_total_input;
		st.start = 0;
		st.len = m_bw_buff_size;
		gpu_cal(st, FN_MOVE_BW);
#else	
		unsigned int   i;

		for (i = 0; i < m_bw_buff_size; i++)
		{
			m_bw_output[i] = m_bw_total_input[i];
		}
#endif
		if (g_config.m_free_type != NERVER_FREE)
		{
			free_bw_input();
			free_next_space();
		}
	}

private:

	// 找到每一个batch的最大值和位置
	void get_max(GTYPE *src, unsigned int   batch, unsigned int   src_num)
	{
		unsigned int   i;
		GTYPE  max = src[0];
		unsigned int   pos = 0;
		GTYPE temp;

		for (i = 1; i < src_num; i++)
		{
			temp = src[i];
			if (temp > max)
			{
				max = temp;
				pos = i;
			}
		}
		m_max_value[batch] = max;
		m_max_pos[batch] = pos;
	}

#ifdef GPU
	void set_const()
	{
		Clayer::set_const();
	}
#endif

	void init_size()
	{
		init_base_size();
		if (m_fw_row == 0 || m_fw_col == 0)
		{
			m_fw_real_row = m_bw_row;
			m_fw_real_col = m_bw_col;
			m_fw_row = m_bw_row;
			m_fw_col = m_bw_col;
		}
		else
		{
			m_fw_real_row = m_fw_row;
			m_fw_real_col = m_fw_col;
		}
		reset_size();
	}

	GTYPE *m_max_value = nullptr;
	unsigned int    *m_max_pos = nullptr;
	GTYPE *m_max_bw = nullptr;
};

#endif