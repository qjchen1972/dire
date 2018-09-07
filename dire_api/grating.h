/*
* Copyright (c) 2017
* All rights reserved.
*
* 文件名称: grating.h
* 文件标识:
* 摘    要: 进行节点数 row col 的转换层。原本是不需要的，后来为了程序看起来思路简单，决定曾加了这一层
*           实现恒等变换 y=x,主要目的就是为了更方便实现残差网络
*
* 当前版本: 1.0
* 作    者: chen.qian.jiang
* 开始日期: 2017-06-28
* 完成日期:
* 其它说明：
* 修改日期        版本号     修改人          修改内容
* -----------------------------------------------
* 2017/06/28      V1.0       陈黔江          创建
* 2017/07/24      V1.0       陈黔江          增加gpu
*/
#pragma once
#ifndef _GRATING__H
#define _GRATING__H

#include "layer.h"

class Grating : public Clayer
{
public:
	Grating() 
	{
		m_layer_type = GRATING_LAYER;
	}
	~Grating() {}

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
		st.result_layer = m_layer_num;
		st.in1 = m_fw_total_input;
		st.start = 0;
		st.len = m_fw_buff_size;
		gpu_cal(st, FN_GRATING_FW);
#else	
		unsigned int   i;
		if (m_grating_type == GRATING_TYPE)
		{
			memcpy(m_fw_output, m_fw_total_input,  sizeof(GTYPE) *m_fw_buff_size);
		}
		else
		{
			int move_x = (m_bw_row - m_fw_row) / 2;
			int move_y = (m_bw_col - m_fw_col) / 2;
			for (i = 0; i < m_fw_buff_size; i++)
			{
				unsigned int   batch_x = i / m_fw_batch_size;
				unsigned int   batch_y = i % m_fw_batch_size;

				unsigned int   node_x = batch_y / m_fw_node_size;
				unsigned int   node_y = batch_y % m_fw_node_size;

				int x = node_y / m_fw_col;
				int y = node_y % m_fw_col;
				m_fw_output[i] = m_fw_total_input[batch_x * m_bw_batch_size + node_x * m_bw_node_size +	(x + move_x) * m_bw_col + y + move_y];
			}
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
#ifdef GPU
		struct gpu_t st;
		st.result = m_bw_output;	
		st.result_layer = m_layer_num;
		st.in1 = m_bw_total_input;
		st.start = 0;
		st.len = m_bw_buff_size;
		gpu_cal(st, FN_GRATING_BW);
#else
		unsigned int   i;
		if (m_grating_type == GRATING_TYPE)
		{
			memcpy(m_bw_output, m_bw_total_input,  sizeof(GTYPE) *m_bw_buff_size);
		}
		else
		{
			int move_x = (m_bw_row - m_fw_row) / 2;
			int move_y = (m_bw_col - m_fw_col) / 2;
			for (i = 0; i < m_bw_buff_size; i++)
			{
				unsigned int   batch_x = i / m_bw_batch_size;
				unsigned int   batch_y = i % m_bw_batch_size;

				unsigned int   node_x = batch_y / m_bw_node_size;
				unsigned int   node_y = batch_y % m_bw_node_size;

				int x = node_y / m_bw_col;
				int y = node_y % m_bw_col;

				if (x < move_x || x >= move_x + m_fw_row || y < move_y || y >= move_y + m_fw_col)
				{
					m_bw_output[i] = 0;					
				}
				else
				{
					m_bw_output[i] = m_bw_total_input[batch_x * m_fw_batch_size + node_x * m_fw_node_size + (x - move_x) * m_fw_col + y - move_y];
				}
			}
		}
#endif
		if (g_config.m_free_type != NERVER_FREE)
		{
			free_bw_input();
			free_next_space();
		}
	}

	void set_type(int type) { m_grating_type = type; }
	
private:

#ifdef GPU
	void set_const()
	{
		g_cpu_layer[m_layer_num].type.grating.grating_type = m_grating_type;
		Clayer::set_const();
	}
#endif
	void init_size()
	{
		init_base_size();

		if (m_grating_type == GRATING_TYPE)
		{
			m_fw_real_row = 1;
			m_fw_real_col = m_bw_node_num* m_bw_row * m_bw_col;
			m_fw_row = m_fw_real_row;
			m_fw_col = m_fw_real_col;
		}
		else if (m_fw_row == 0 || m_fw_col == 0)
		{
			m_fw_real_row = m_bw_row;
			m_fw_real_col = m_bw_col;
			m_fw_row = m_fw_real_row;
			m_fw_col = m_fw_real_col;
		}
		else
		{
			m_fw_real_row = m_fw_row;
			m_fw_real_col = m_fw_col;
		}
		if (m_grating_type == GRATING_TYPE) m_fw_node_num = 1;
		reset_size();
	}

	int  m_grating_type = CENTRE_TYPE;
};

#endif