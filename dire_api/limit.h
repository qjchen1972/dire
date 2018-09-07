#pragma once
/*
* Copyright (c) 2017
* All rights reserved.
*
* 文件名称: limit.h
* 文件标识:
* 摘    要: 把超出最大和最小值的输入进行消减，避免softmax和cross-entropy溢出
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
#ifndef _LIMIT__H
#define _LIMIT__H

#include "layer.h"

class Limit : public Clayer
{
public:
	Limit()
	{
		m_layer_type = LIMIT_LAYER;
	}
	~Limit() {}


	void init_space()
	{
#ifdef GPU
		set_const();		
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
		st.result = m_fw_output;
		st.in1 = m_fw_total_input;
		st.start = 0;
		st.len = m_fw_buff_size;
		gpu_cal(st, FN_LIMIT_FW);
#else
		unsigned int   i;
		
		for (i = 0; i < m_fw_buff_size; i++)
		{
			if (m_fw_total_input[i] > LIMIT_MAX)
			{
				m_fw_output[i] = LIMIT_MAX;
			}
			else if (m_fw_total_input[i] < LIMIT_MIN)
			{
				m_fw_output[i] = LIMIT_MIN;
			}
			else
				m_fw_output[i] = m_fw_total_input[i];
		}
#endif
		if (g_config.m_free_type == FUNC_FREE)	free_fw_input();
		
	}

	void backword_proc()
	{
		if (m_next_num <= 0) return;
		if (!m_fw_total_input)
		{
			create_fw_input();
			get_fw_input();
		}
		create_bw_input();
		get_bw_input();
		create_bw_output();
		
#ifdef GPU 
		struct gpu_t st;
		st.result = m_bw_output;
		st.in1 = m_fw_total_input;
		st.in2 = m_bw_total_input;
		st.start = 0;
		st.len = m_bw_buff_size;
		gpu_cal(st, FN_LIMIT_BW);
#else
		unsigned int   i;
		for (i = 0; i < m_bw_buff_size; i++)
		{		
			if (m_fw_total_input[i] > LIMIT_MAX)
				m_bw_output[i] = 0;
			else if (m_fw_total_input[i] < LIMIT_MIN)
				m_bw_output[i] = 0;
			else
				m_bw_output[i] = m_bw_total_input[i];
		}
#endif
		if (g_config.m_free_type != NERVER_FREE)
		{
			free_fw_input();
			free_bw_input();
			free_prev_space();
			free_next_space();
		}
	}

private:

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
};

#endif