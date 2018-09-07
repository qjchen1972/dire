/*
* Copyright (c) 2017
* All rights reserved.
*
* 文件名称: batchnorm.h
* 文件标识:
* 摘    要: 数据进行z-score处理，按照中心极限，当n充分大时，数据趋近正态分布
*           需要注意的事，采用BN的处理，会导致bias失效，所以通常采用bn时，后跟一个scale层
*
* 当前版本: 1.0
* 作    者: chen.qian.jiang
* 开始日期: 2017-06-18
* 完成日期:
* 其它说明：
* 修改日期        版本号     修改人          修改内容
* -----------------------------------------------
* 2017/06/21      V1.0       陈黔江          创建
* 2017/07/12      V1.0       陈黔江          增加gpu
*/

#pragma once
#ifndef _BATCHNORM__H
#define _BATCHNORM__H


#include "layer.h"

class Batchnorm : public Clayer
{
public:
	Batchnorm() 
	{
		m_layer_type = BN_LAYER;		
	}
	~Batchnorm() {}

	void init_space()
	{
#ifdef GPU
		set_const();
		HANDLE_ERROR(cudaMalloc((void **)&m_ex_gpu, sizeof(GTYPE)*m_fw_node_num));
		HANDLE_ERROR(cudaMalloc((void **)&m_var_gpu, sizeof(GTYPE)*m_fw_node_num));
		HANDLE_ERROR(cudaMalloc((void **)&m_bw_ex, sizeof(GTYPE)*m_fw_node_num));
		HANDLE_ERROR(cudaMalloc((void **)&m_bw_multex, sizeof(GTYPE)*m_fw_node_num));
#else
		//用于后向计算的临时buf
		m_bw_ex = new GTYPE[m_fw_node_num];
		m_bw_multex = new GTYPE[m_fw_node_num];
#endif
	}

	void init()
	{
		init_size();

		m_ex = new GTYPE[m_fw_node_num];
		m_var = new GTYPE[m_fw_node_num];
		m_total_ex = new GTYPE[m_fw_node_num];
		m_total_var = new GTYPE[m_fw_node_num];

		memset(m_total_ex, 0, m_fw_node_num * sizeof(GTYPE));
		memset(m_total_var, 0, m_fw_node_num * sizeof(GTYPE));
	}

	void init_param(ifstream *f = nullptr)
	{
		/*sprintf(m_param_file, "%s%d.txt", BN_PARAM_FILE, m_layer_num);
		ifstream ff;
		char file[FILENAME_LEN];
		sprintf(file, "%s/%s", g_config.m_work_param_dir, m_param_file);
		if (!open_in_file(ff, file)) return ;
		int i;
		for (i = 0; i < m_fw_node_num; i++)
		{
			ff >> m_total_ex[i] >> m_total_var[i];
		}
		ff.close();
		return;
		*/

		if (m_net_status == TESTING || m_net_status == CONT_TRAINING)
		{
			if (read_param(f)) return;
		}
	}

	void save_param(ofstream *f)
	{
		write_param(f);
	}

	void forward_proc()
	{
		if (m_prev_num <= 0) return;
		create_fw_input();
		get_fw_input();		
		create_fw_output();
	
		unsigned int   i, j;
		unsigned int    elv_num = m_batch_num * m_bw_node_size;

		//最开始需要对全局期望和方差清0
		//若是测试，把统计值赋入
		if (m_net_status == TESTING )
		{
			if (g_config.m_test_mode == GLOBAL_EX)
			{
				unsigned int train_num = g_config.m_train_batch_num * m_bw_node_size;
				GTYPE temp = 1.0*train_num / (train_num - 1);				
				for (i = 0; i < m_bw_node_num; i++)
				{
					m_ex[i] = m_total_ex[i];
					m_var[i] = m_total_var[i] * temp;
				}
			}
			else
			{
				GTYPE *fw_total_input = new GTYPE[m_bw_buff_size];
#ifdef GPU	
				HANDLE_ERROR(cudaMemcpy(fw_total_input, m_fw_total_input, sizeof(GTYPE)*m_bw_buff_size, cudaMemcpyDeviceToHost));
#endif
				for (i = 0; i < m_bw_node_num; i++)
				{
					m_ex[i] = 0;
					for (j = 0; j < elv_num; j++)
					{
						unsigned int    batch_x = j / m_bw_node_size;
						unsigned int   batch_y = j % m_bw_node_size;
						m_ex[i] += fw_total_input[batch_x *  m_bw_batch_size + i*m_bw_node_size + batch_y];
					}
					m_ex[i] = m_ex[i] / elv_num;

					m_var[i] = 0;
					for (j = 0; j < elv_num; j++)
					{
						unsigned int   batch_x = j / m_bw_node_size;
						unsigned int   batch_y = j %  m_bw_node_size;
						unsigned int   pos = batch_x *  m_bw_batch_size + i*m_bw_node_size + batch_y;
						m_var[i] += (fw_total_input[pos] - m_ex[i]) * (fw_total_input[pos] - m_ex[i]);
					}
					m_var[i] = m_var[i] / elv_num;
				}
				delete[] fw_total_input;
			}
		}		

#ifdef GPU		
		struct gpu_t st;
		if (m_net_status != TESTING)
		{			
			st.result = m_fw_total_input;
			st.result_layer = m_layer_num;
			st.start = 0;
			st.len = elv_num;
			st.sum_result = m_ex_gpu;
			st.sum_result_count = m_bw_node_num;
			gpu_sum(st, FN_BN_FW_EX_SUM);
						
			st.result = m_fw_total_input;
			st.result_layer = m_layer_num;
			st.in1 = m_ex_gpu;
			st.start = 0;
			st.len = elv_num;
			st.sum_result = m_var_gpu;
			st.sum_result_count = m_bw_node_num;
			gpu_sum(st, FN_BN_FW_VAR_SUM);
		
			HANDLE_ERROR(cudaMemcpy(m_ex, m_ex_gpu, sizeof(GTYPE)*m_fw_node_num, cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(m_var, m_var_gpu, sizeof(GTYPE)*m_fw_node_num, cudaMemcpyDeviceToHost));		
		}
		else
		{
			HANDLE_ERROR(cudaMemcpy(m_ex_gpu, m_ex, sizeof(GTYPE)*m_fw_node_num, cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(m_var_gpu, m_var, sizeof(GTYPE)*m_fw_node_num, cudaMemcpyHostToDevice));
		}

		st.result = m_fw_output;
		st.result_layer = m_layer_num;
		st.in1 = m_fw_total_input;
		st.in2 = m_ex_gpu;
		st.in3 = m_var_gpu;
		st.start = 0;
		st.len = m_fw_buff_size;
		gpu_cal(st, FN_BN_FW);
#else			
		
		if (m_net_status != TESTING)
		{
			for (i = 0; i < m_bw_node_num; i++)
			{
				m_ex[i] = 0;
				for (j = 0; j < elv_num; j++)
				{
					unsigned int    batch_x = j / m_bw_node_size;
					unsigned int   batch_y = j % m_bw_node_size;
					m_ex[i] += m_fw_total_input[batch_x *  m_bw_batch_size + i*m_bw_node_size + batch_y];
				}
				m_ex[i] = m_ex[i] / elv_num;

				m_var[i] = 0;
				for (j = 0; j < elv_num; j++)
				{
					unsigned int   batch_x = j / m_bw_node_size;
					unsigned int   batch_y = j %  m_bw_node_size;
					unsigned int   pos = batch_x *  m_bw_batch_size + i*m_bw_node_size + batch_y;
					m_var[i] += (m_fw_total_input[pos] - m_ex[i]) * (m_fw_total_input[pos] - m_ex[i]);
				}
				m_var[i] = m_var[i] / elv_num;
			}
		}

		for (i = 0; i < m_fw_buff_size; i++)
		{
			unsigned int   batch_y = i % m_fw_batch_size;
			unsigned int   node_x = batch_y / m_fw_node_size;
			m_fw_output[i] = (m_fw_total_input[i] - m_ex[node_x]) / sqrt(m_var[node_x] + INFINIT_NUM);			
		}
#endif


		if (m_net_status != TESTING)
		{
			//最开始需要对全局期望和方差清0，下面这种方式保证了m_train_num=1时的清0
			for (i = 0; i < m_bw_node_num; i++)
			{
				m_total_ex[i] = (m_total_ex[i] * (m_train_num - 1) + m_ex[i]) / m_train_num;
				m_total_var[i] = (m_total_var[i] * (m_train_num - 1) + m_var[i]) / m_train_num;
			}
		}

		if (g_config.m_free_type != NERVER_FREE)
		{
			free_fw_input();
			free_prev_space();
			m_fw_del_request--;
		}
	}

	//mean(x)表示x的期望值
	// if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
	// dE(Y)/dX = (dE/dY - mean(dE/dY) - mean(dE/dY *Y) *Y)/ sqrt(var(X) + eps)
	//mean表示的是m_batch_num*m_node_size 这个块的均值
	void backword_proc()
	{
		if (m_next_num <= 0 ) return;
		create_bw_input();
		get_bw_input();
		create_bw_output();
	
		
		unsigned int   elv_num = m_batch_num * m_fw_node_size;
#ifdef GPU		
		struct gpu_t st;
		st.result = m_bw_total_input;
		st.result_layer = m_layer_num;
		st.start = 0;
		st.len = elv_num;
		st.sum_result = m_bw_ex;
		st.sum_result_count = m_fw_node_num;
		gpu_sum(st, FN_BN_BW_EX_SUM);


		st.result = m_bw_total_input;
		st.result_layer = m_layer_num;
		st.in1 = m_fw_output;
		st.start = 0;
		st.len = elv_num;
		st.sum_result = m_bw_multex;
		st.sum_result_count = m_fw_node_num;
		gpu_sum(st, FN_BN_BW_EXMULTI_SUM);

		st.result = m_bw_output;
		st.result_layer = m_layer_num;
		st.in1 = m_bw_total_input;
		st.in2 = m_fw_output;
		st.in3 = m_bw_ex;
		st.in4 = m_bw_multex;
		st.sum_result = m_var_gpu;
		st.start = 0;
		st.len = m_bw_buff_size;
		gpu_cal(st, FN_BN_BW);

#else		
		unsigned int   i;		
		for (i = 0; i < m_fw_buff_size; i++)
		{
			unsigned int    node_x = i / elv_num;
			unsigned int    node_y = i % elv_num;

			if (node_y == 0)
			{
				m_bw_ex[node_x] = 0;
				m_bw_multex[node_x] = 0;
			}
			unsigned int   batch_x = node_y / m_fw_node_size;
			unsigned int   batch_y = node_y % m_fw_node_size;
			unsigned int   pos = batch_x * m_fw_batch_size + node_x*m_fw_node_size + batch_y;

			m_bw_ex[node_x] += m_bw_total_input[pos] / elv_num;
			m_bw_multex[node_x] += m_bw_total_input[pos] * m_fw_output[pos] / elv_num;
		}
	
		for (i = 0; i < m_bw_buff_size; i++)
		{
			unsigned int   batch_y = i % m_bw_batch_size;
			unsigned int   node_x = batch_y / m_bw_node_size;
			m_bw_output[i] = (m_bw_total_input[i] - m_bw_ex[node_x] - m_bw_multex[node_x] * m_fw_output[i]) / sqrt(m_var[node_x] + INFINIT_NUM);
		}

#endif

		if (g_config.m_free_type != NERVER_FREE)
		{
			free_bw_input();
			free_next_space();
			free_fw_output();
		}
	}		


private:	

	bool write_param(ofstream *f)
	{
		(*f).write((char*)m_total_ex, sizeof(GTYPE)*m_fw_node_num);
		(*f).write((char*)m_total_var, sizeof(GTYPE)*m_fw_node_num);
		return true;
	}

	bool read_param(ifstream *f)
	{
		(*f).read((char*)m_total_ex, sizeof(GTYPE)*m_fw_node_num);
		if ((*f).eof()) return false;
		(*f).read((char*)m_total_var, sizeof(GTYPE)*m_fw_node_num);
		if ((*f).eof()) return false;
		return true;
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

	
	GTYPE *m_total_ex = nullptr; //全局的期望值
	GTYPE *m_total_var = nullptr;// 全局的方差值
	GTYPE *m_ex = nullptr; // 期望
	GTYPE *m_var = nullptr;// 方差
	//用于后向计算的临时期望
	GTYPE *m_bw_ex = nullptr;
	GTYPE *m_bw_multex = nullptr;

#ifdef GPU	
	GTYPE *m_ex_gpu = nullptr; //期望值
	GTYPE *m_var_gpu = nullptr;//方差值		
#endif
};

#endif
