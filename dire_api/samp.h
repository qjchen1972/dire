/*
* Copyright (c) 2017
* All rights reserved.
*
* 文件名称: samp.h
* 文件标识:
* 摘    要: 池化层
*
*
* 当前版本: 1.0
* 作    者: chen.qian.jiang
* 开始日期: 2017-06-29
* 完成日期:
* 其它说明：
* 修改日期        版本号     修改人          修改内容
* -----------------------------------------------
* 2017/06/29      V1.0       陈黔江          创建
* 2017/06/27      V1.0       陈黔江          增加gpu
*/
#pragma once
#ifndef _SAMP__H
#define _SAMP__H

#include "layer.h"

class Samp :public Clayer
{
public:
	Samp() 
	{
		m_layer_type = SAMP_LAYER;		
	}

	~Samp(){}


	void init_space()
	{
#ifdef GPU
		set_const();
		HANDLE_ERROR(cudaMalloc((void **)&m_weight_gpu, sizeof(GTYPE) *m_fw_node_num));
		HANDLE_ERROR(cudaMalloc((void **)&m_weight_diff_gpu, sizeof(GTYPE) *m_fw_node_num));
		if (m_biasinuse)
		{
			HANDLE_ERROR(cudaMalloc((void **)&m_b_gpu, sizeof(GTYPE) *m_fw_node_num));
			HANDLE_ERROR(cudaMalloc((void **)&m_b_diff_gpu, sizeof(GTYPE) *m_fw_node_num));
	    }
#endif
		if (m_pool_type == MAX_POOL)
		{
#ifdef GPU
			HANDLE_ERROR(cudaMalloc((void **)&m_max_pos, sizeof(unsigned int  )*m_bw_buff_size));
#else
			m_max_pos = new unsigned int  [m_bw_buff_size];
#endif
		}
	}

	void init()
	{
		init_size();

		m_weight = new GTYPE[m_fw_node_num];		
		m_weight_diff = new GTYPE[m_fw_node_num];

		if (m_biasinuse)
		{
			m_b = new GTYPE[m_fw_node_num];
			m_b_diff = new GTYPE[m_fw_node_num];
		}

		if (m_net_status == PARAM_CHECK)
		{
			m_weight_check = new GTYPE[m_fw_node_num];
			if (m_biasinuse) m_b_check = new GTYPE[m_fw_node_num];
		}
		
		if (g_config.m_study_mode == ADAM_STUDY)
		{
			m_weight_one = new GTYPE[m_fw_node_num];			
			m_weight_two = new GTYPE[m_fw_node_num];
			if (m_biasinuse)
			{
				m_b_one = new GTYPE[m_fw_node_num];
				m_b_two = new GTYPE[m_fw_node_num];
			}
		}
		if (g_config.m_study_mode == MOMENTUM_STUDY)
		{
			m_weight_diff_before = new GTYPE[m_fw_node_num];
			if (m_biasinuse) m_b_diff_before = new GTYPE[m_fw_node_num];
		}
	}

	void init_param(ifstream *f = nullptr)
	{
		/*sprintf(m_param_file, "%s%d.txt", SAMP_PARAM_FILE, m_layer_num);
		ifstream ff;
		char file[FILENAME_LEN];
		sprintf(file, "%s/%s", g_config.m_work_param_dir, m_param_file);
		if (!open_in_file(ff, file)) return;
		for (int i = 0; i < m_fw_node_num; i++)
		{		
			ff >> m_weight[i] >> m_b[i];
		}
		ff.close();
		return;
		*/

		if (m_net_status == PARAM_CHECK)
			sprintf(m_check_file, "%s%d.txt", SAMP_CHECK_LOG, m_layer_num);

		if (m_net_status == CONT_TRAINING || m_net_status == TESTING)
		{
			read_param(f);
			return;
		}

		//初始化weight
		for (int i = 0; i < m_fw_node_num; i++)
		{
			if (m_biasinuse)
			{
				//bias 只需要方差 -> 0
				if (g_config.m_initweight_mode == XAVIER)
				{
					m_b[i] = uniform_rand(-sqrt(3 * INFINIT_NUM), sqrt(3 * INFINIT_NUM));
				}
				else if (g_config.m_initweight_mode == MARS)
				{
					m_b[i] = gauss_rand(0, INFINIT_NUM);
				}
			}
			//weight 
			//若是前层是relu，var(yk) = 1/2*var(w)var(yk-1) + var(b) 
			// var(w) = 2
			//若是leakyrelu, var(yk) = 1/2 *(1+a*a) * var(yk-1) * var(w) + var(b)
			// var(w) = 2/(1+a*a)
			//若是其他，var(yk) = var(w)var(yk-1) + var(b)
			// var(w) = 1
			if (g_config.m_initweight_mode == XAVIER)
			{
				if (!strcmp(m_prev_layer[0]->m_layer_type, ACIVE_LAYER))
				{
					int  type = dynamic_cast<Active*>(m_prev_layer[0])->get_atcive_type();
					if (type == RELU)
					{
						m_weight[i] = uniform_rand(-sqrt(6), sqrt(6));
					}
					else if (type == LEAKY_RELU)
					{
						GTYPE  t7 = 1.0 / (1 + dynamic_cast<Active*>(m_prev_layer[0])->get_leakey_relu()*dynamic_cast<Active*>(m_prev_layer[0])->get_leakey_relu());
						GTYPE  t8 = sqrt(6.0*t7);
						m_weight[i] = uniform_rand(-t8, t8);
					}
					else
					{
						m_weight[i] = uniform_rand(-sqrt(3), sqrt(3));
					}
				}
				else
				{
					m_weight[i] = uniform_rand(-sqrt(3), sqrt(3));
				}
			}
			else if (g_config.m_initweight_mode == MARS)
			{
				if (!strcmp(m_prev_layer[0]->m_layer_type, ACIVE_LAYER))
				{
					int  type = dynamic_cast<Active*>(m_prev_layer[0])->get_atcive_type();
					if (type == RELU)
					{
						m_weight[i] = gauss_rand(0, 2);
					}
					else if (type == LEAKY_RELU)
					{
						GTYPE  t7 = 1 / (1 + dynamic_cast<Active*>(m_prev_layer[0])->get_leakey_relu()*dynamic_cast<Active*>(m_prev_layer[0])->get_leakey_relu());
						GTYPE  t8 = 2.0*t7;
						m_weight[i] = gauss_rand(0, t8);
					}
					else
					{
						m_weight[i] = gauss_rand(0, 1);
					}
				}
				else
				{
					m_weight[i] = gauss_rand(0, 1);
				}
			}
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
		set_param();

#ifdef GPU
		struct gpu_t st;
		st.result = m_fw_output;
		st.result_layer = m_layer_num;
		st.in1 = m_fw_total_input;
		st.in2 = m_weight_gpu;
		if (m_biasinuse)	st.in3 = m_b_gpu;
		if(m_pool_type == MAX_POOL)
			st.in5 = m_max_pos;
		st.start = 0;
		st.len = m_fw_buff_size;
		gpu_cal(st, FN_SAMP_FW);

#else		
		unsigned int   i;
		for (i = 0; i < m_fw_buff_size; i++)
		{
			unsigned int   batch_x = i / m_fw_batch_size;
			unsigned int   batch_y = i % m_fw_batch_size;
			unsigned int   node_x = batch_y / m_fw_node_size;
			unsigned int   node_y = batch_y % m_fw_node_size;

			int u = node_y / m_fw_col;
			int v = node_y % m_fw_col;
			if (u < m_top_padding || u >= m_fw_row - m_bottom_padding || v < m_left_padding || v >= m_fw_col - m_right_padding)
			{
				m_fw_output[i] = 0;
				continue;
			}
			unsigned int    src_pos = batch_x* m_bw_batch_size + node_x * m_bw_node_size;
			int src_x = (u - m_top_padding)*m_row_stride;
			int src_y = (v - m_left_padding)*m_col_stride;
			if (m_biasinuse)  
				m_fw_output[i] = sample(src_pos, src_x, src_y) * m_weight[node_x] + m_b[node_x];
			else
				m_fw_output[i] = sample(src_pos, src_x, src_y) * m_weight[node_x];
		}
#endif
		if (g_config.m_free_type == FUNC_FREE)	free_fw_input();
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
		st.in2 = m_weight_gpu;		
		if (m_pool_type == MAX_POOL)
			st.in5 = m_max_pos;
		st.start = 0;
		st.len = m_bw_buff_size;
		gpu_cal(st, FN_SAMP_BW);
#else
		unsigned int   i;
		for (i = 0; i < m_bw_buff_size; i++)
		{
			unsigned int   batch_x = i / m_bw_batch_size;
			unsigned int   batch_y = i % m_bw_batch_size;
			unsigned int   node_x = batch_y / m_bw_node_size;
			unsigned int   node_y = batch_y % m_bw_node_size;

			int u = node_y / m_bw_col;
			int v = node_y % m_bw_col;

			unsigned int   src_pos = batch_x*m_bw_batch_size + node_x * m_bw_node_size;
			unsigned int    loss_pos = batch_x*m_fw_batch_size + node_x * m_fw_node_size;
			m_bw_output[i] = get_bw_pool_matrix_one(src_pos, u, v, loss_pos) * m_weight[node_x];
		}
#endif
		if (g_config.m_free_type == FUNC_FREE)  free_bw_input();
	}			

	//bias的导数是= sigma(dE/dY) 
	// weight的导数是sigma(dE/dY*f(PP(a)))， y = f(PP(x)) ,f函数是池化方式。PP是集合的表示，俺自己定义的,表示{x | x 属于A}
	void update_weight()
	{
		unsigned int   i;
		get_diff();
		for (i = 0; i < m_fw_node_num; i++)
		{
			if (g_config.m_study_mode == ADAM_STUDY)
			{
				if (m_biasinuse)	adam_study(m_b_diff[i], &m_b_one[i], &m_b_two[i], &m_b[i]);
				adam_study(m_weight_diff[i], &m_weight_one[i], &m_weight_two[i], &m_weight[i]);
			}
			else if (g_config.m_study_mode == MOMENTUM_STUDY)
			{
				if (m_biasinuse)	momentum_study(m_b_diff[i], &m_b[i], &m_b_diff_before[i]);
				momentum_study(m_weight_diff[i], &m_weight[i], &m_weight_diff_before[i]);
			}
			else
			{
				if (m_biasinuse)	static_study(m_b_diff[i], &m_b[i]);
				static_study(m_weight_diff[i], &m_weight[i]);
			}
		}
	}		

	void create_test_weight(GTYPE(*f)(Clayer *start), Clayer *start)
	{
		GTYPE temp;
		GTYPE err1, err2;
		unsigned int   i;
		for ( i = 0; i < m_fw_node_num; i++)
		{
			if (m_biasinuse)
			{
				temp = m_b[i];
				m_b[i] = temp - INFINIT_NUM;
				err1 = f(start);
				m_b[i] = temp + INFINIT_NUM;
				err2 = f(start);
				m_b_check[i] = (err2 - err1 + regular_check(temp)) / (2.0 * INFINIT_NUM);
				m_b[i] = temp;
			}

			temp = m_weight[i];
			m_weight[i] = temp - INFINIT_NUM;
			err1 = f(start);
			m_weight[i] = temp + INFINIT_NUM;
			err2 = f(start);		
			m_weight_check[i] = (err2 - err1 + regular_check(temp)) / (2.0 * INFINIT_NUM);
			m_weight[i] = temp;
		}
	}

	void check_weight()
	{
		ofstream f;
		if (!open_out_file(f, m_check_file)) return;
				
		unsigned int   i;
		get_diff();
		f << setprecision(PARAM_PRECISION) << endl;
		for (i = 0; i < m_fw_node_num; i++)
		{
			if (m_biasinuse)
				f << " b_check[" << i << "] " << m_b_check[i] << " " << regular_proc(m_b_diff[i], m_b[i]);						
			f << "   weight_check[" << i << "]  " << m_weight_check[i] << " " << regular_proc(m_weight_diff[i], m_weight[i]) << endl;
		}		
		f.close();
	}

	void set_param(int type, int samp_row, int samp_col, int row_stride, int col_stride)
	{
		m_pool_type = type;
		m_pool_row = samp_row;
		m_pool_col = samp_col;
		m_row_stride = row_stride;
		m_col_stride = col_stride;
	}

	void close_bias() { m_biasinuse = false; }

private:
	
	////池化计算
	GTYPE sample(unsigned int    src_pos, int src_x, int src_y)
	{
		GTYPE temp = 0;
		int m, n;

		if (m_pool_type == MAX_POOL)
		{
			/*
			  max_pos表示的是每一个池化左上角位置的标注，值是这个池化最大位置的位置
			  所以寻找池化后的值在前向输入矩阵的位置，首先要转化为前向矩阵所在的最左上角的位置p,m_max_pos[p]就是最大值的位置
			*/
			
			temp = m_fw_total_input[src_pos + src_x * m_bw_col + src_y];
			m_max_pos[src_pos + src_x * m_bw_col + src_y] = src_pos + src_x * m_bw_col + src_y;

			for (m = 0; m < m_pool_row; m++)
				for (n = 0; n < m_pool_col; n++)
				{
					if (src_x + m < m_bw_row && src_y + n < m_bw_col)
					{						
						if ( m_fw_total_input[src_pos + (src_x + m) * m_bw_col + src_y + n] > temp )
						{
							temp = m_fw_total_input[src_pos + (src_x + m) * m_bw_col + src_y + n];
							m_max_pos[src_pos + src_x * m_bw_col + src_y] = src_pos + (src_x + m) * m_bw_col + src_y + n;
						}
					}
				}
		}
		else if (m_pool_type == AVG_POOL || m_pool_type == GLOBAL_AVG_POOL)
		{
			for (m = 0; m < m_pool_row; m++)
				for (n = 0; n < m_pool_col; n++)
				{
					if (src_x + m < m_bw_row && src_y + n < m_bw_col)
					{
						temp += m_fw_total_input[src_pos + (src_x + m) * m_bw_col + src_y + n];
					}
				}
			temp = temp / (m_pool_row * m_pool_col);
		}
		else
			temp = 0;
		return temp;
	}
	
	// 后向，此时是有padding和stride, (u,v)为前向输入矩阵的位置坐标 
	//具体推导 可见博客
	GTYPE get_bw_pool_matrix_one(unsigned int   src_pos, int u,int v, unsigned int   loss_pos)
	{
		GTYPE answer = 0;
		int m, n;
		for (m = 0; m < m_pool_row; m++)
			for (n = 0; n < m_pool_col; n++)
			{
				if (u - m >= 0 && v - n >= 0 && (u - m) / m_row_stride < m_fw_real_row && (v - n) / m_col_stride < m_fw_real_col &&
					(u - m) % m_row_stride == 0 && (v - n) % m_col_stride == 0)
				{
					if (m_pool_type == MAX_POOL)
					{
						int src_x = u - m;
						int src_y = v - n;
						GTYPE drev = 0;
						if (m_max_pos[src_pos + src_x*m_bw_col + src_y] == src_pos + u*m_bw_col + v) drev = 1;
						answer += m_bw_total_input[loss_pos + ((u - m) / m_row_stride + m_top_padding)*m_fw_col + (v - n) / m_col_stride + m_left_padding] * drev;
					}
					else if (m_pool_type == AVG_POOL || m_pool_type == GLOBAL_AVG_POOL)
					{
						GTYPE drev = 1.0 / (m_pool_row*m_pool_col);
						answer += m_bw_total_input[loss_pos + ((u - m) / m_row_stride + m_top_padding)*m_fw_col + (v - n) / m_col_stride + m_left_padding] * drev;
					
					}
					else
						answer = 0;
				}
			}
		return answer;
	}
	
	bool write_param(ofstream *f)
	{
		if (m_biasinuse) (*f).write((char*)m_b, sizeof(GTYPE)*m_fw_node_num);
		(*f).write((char*)m_weight, sizeof(GTYPE)* m_fw_node_num);
		return true;
	}

	bool read_param(ifstream *f)
	{
		if (m_biasinuse) (*f).read((char*)m_b, sizeof(GTYPE)*m_fw_node_num);
		if ((*f).eof()) return false;
		(*f).read((char*)m_weight, sizeof(GTYPE)* m_fw_node_num);
		if ((*f).eof()) return false;
		return true;
	}

#ifdef GPU
	void set_const()
	{
		g_cpu_layer[m_layer_num].type.samp.top_padding = m_top_padding;
		g_cpu_layer[m_layer_num].type.samp.bottom_padding = m_bottom_padding;
		g_cpu_layer[m_layer_num].type.samp.left_padding = m_left_padding;
		g_cpu_layer[m_layer_num].type.samp.right_padding = m_right_padding;
		g_cpu_layer[m_layer_num].type.samp.row_stride = m_row_stride;
		g_cpu_layer[m_layer_num].type.samp.col_stride = m_col_stride;
		g_cpu_layer[m_layer_num].type.samp.pool_type = m_pool_type;
		g_cpu_layer[m_layer_num].type.samp.pool_row = m_pool_row;
		g_cpu_layer[m_layer_num].type.samp.pool_col = m_pool_col;
		Clayer::set_const();
	}
#endif

	void  set_param()
	{
#ifdef GPU
		if (m_biasinuse)
			HANDLE_ERROR(cudaMemcpy(m_b_gpu, m_b, sizeof(GTYPE) *m_fw_node_num, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(m_weight_gpu, m_weight, sizeof(GTYPE) *m_fw_node_num, cudaMemcpyHostToDevice));
#endif
	}

	void init_size()
	{
		init_base_size();

		if (m_pool_type == GLOBAL_AVG_POOL)
		{
			m_pool_row = m_bw_row;
			m_pool_col = m_bw_col;
			m_row_stride = 1;
			m_col_stride = 1;
		}

		//计算fw实际的大小	
		m_fw_real_row = (m_bw_row - m_pool_row) / m_row_stride + 1;
		m_fw_real_col = (m_bw_col - m_pool_col) / m_col_stride + 1;
		if ((m_bw_row - m_pool_row) % m_row_stride != 0)
		{
			if ((m_bw_row - m_pool_row) % m_row_stride + m_pool_row  > m_row_stride)
				m_fw_real_row += 1;
		}
		if ((m_bw_col - m_pool_col) % m_col_stride != 0)
		{
			if ((m_bw_col - m_pool_col) % m_col_stride + m_pool_col  > m_col_stride)
				m_fw_real_col += 1;
		}

		if (m_fw_row == 0)
		{
			m_fw_row = m_fw_real_row;
			m_fw_col = m_fw_real_col;
		}
		else
		{
			if (m_fw_row < m_fw_real_row) m_fw_row = m_fw_real_row;
			if (m_fw_col < m_fw_real_col) m_fw_col = m_fw_real_col;

			//计算padding
			if ((m_fw_row - m_fw_real_row) % 2 == 0)
			{
				m_top_padding = m_bottom_padding = (m_fw_row - m_fw_real_row) / 2;
			}
			else
			{
				m_top_padding = (m_fw_row - m_fw_real_row) / 2;
				m_bottom_padding = (m_fw_row - m_fw_real_row) / 2 + 1;
			}
			if ((m_fw_col - m_fw_real_col) % 2 == 0)
			{
				m_left_padding = m_right_padding = (m_fw_col - m_fw_real_col) / 2;
			}
			else
			{
				m_left_padding = (m_fw_col - m_fw_real_col) / 2;
				m_right_padding = (m_fw_col - m_fw_real_col) / 2 + 1;
			}
		}
		reset_size();
	}
	
	void get_diff()
	{		
		unsigned int   nodes_len = m_batch_num * m_fw_row * m_fw_col;

		if (!m_bw_total_input)
		{
			create_bw_input();
			get_bw_input();
		}
		if (!m_fw_total_input)
		{
			create_fw_input();
			get_fw_input();
		}

#ifdef GPU
		
		struct gpu_t st;
		if (m_biasinuse)
		{			
			st.result = m_bw_total_input;
			st.result_layer = m_layer_num;
			st.start = 0;
			st.len = nodes_len;
			st.sum_result = m_b_diff_gpu;
			st.sum_result_count = m_fw_node_num;
			st.batchsize = m_batch_num;
			gpu_sum(st, FN_SAMP_UPDATE_B_SUM);
		}

		st.result = m_bw_total_input;
		st.result_layer = m_layer_num;
		st.in1 = m_fw_total_input;
		if (m_pool_type == MAX_POOL)
			st.in5 = m_max_pos;
		st.start = 0;
		st.len = nodes_len;
		st.sum_result = m_weight_diff_gpu;
		st.sum_result_count = m_fw_node_num;
		st.batchsize = m_batch_num;
		gpu_sum(st, FN_SAMP_UPDATE_WEIGHT_SUM);		

		if (m_biasinuse)
			HANDLE_ERROR(cudaMemcpy(m_b_diff, m_b_diff_gpu, sizeof(GTYPE) *m_fw_node_num, cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(m_weight_diff, m_weight_diff_gpu, sizeof(GTYPE) *m_fw_node_num, cudaMemcpyDeviceToHost));

#else
		unsigned int   i, j;
		for (i = 0; i < m_fw_node_num; i++)
		{
			m_weight_diff[i] = 0;
			if (m_biasinuse) m_b_diff[i] = 0;
			for (j = 0; j < nodes_len; j++)
			{
				unsigned int   batch_x = j / m_fw_node_size;
				unsigned int   batch_y = j % m_fw_node_size;

				int x = batch_y / m_fw_col;
				int y = batch_y % m_fw_col;

				if (x < m_top_padding || x >= m_fw_row - m_bottom_padding || y < m_left_padding || y >= m_fw_col - m_right_padding)
					continue;

				unsigned int   pos = batch_x * m_fw_batch_size + i*m_fw_node_size + batch_y;

				if (m_biasinuse)
					m_b_diff[i] += m_bw_total_input[pos];

				unsigned int    src_pos = batch_x * m_bw_batch_size + i*m_bw_node_size;
				int src_x = (x - m_top_padding)*m_row_stride;
				int src_y = (y - m_left_padding)*m_col_stride;
				m_weight_diff[i] += m_bw_total_input[pos] * sample(src_pos, src_x, src_y);
			}
			if (m_biasinuse)		
				m_b_diff[i] = m_b_diff[i] / m_batch_num;
			m_weight_diff[i] = m_weight_diff[i] / m_batch_num;
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

	int  m_row_stride = 1;
	int  m_col_stride = 1;
	int  m_top_padding = 0;
	int  m_bottom_padding = 0;
	int  m_left_padding = 0;
	int  m_right_padding = 0;

	//权值
	GTYPE  *m_weight = nullptr;
	GTYPE  *m_b = nullptr;

	//用于梯度校验
	GTYPE  *m_weight_diff = nullptr;
	GTYPE  *m_b_diff = nullptr;

	GTYPE  *m_weight_check = nullptr;
	GTYPE  *m_b_check = nullptr;

	// Adam学习
	GTYPE  *m_weight_one = nullptr;
	GTYPE  *m_b_one = nullptr;
	GTYPE  *m_weight_two = nullptr;
	GTYPE  *m_b_two = nullptr;
	//冲量学习
	GTYPE  *m_weight_diff_before = nullptr;
	GTYPE  *m_b_diff_before = nullptr;


	//池化矩阵的大小，全局池化时，行和列与输入矩阵相同
	int  m_pool_row = 0;
	int  m_pool_col = 0;
	//池化的类型
	int  m_pool_type = AVG_POOL;
	//若是最大池化，记录最大值得位置
	unsigned int   *m_max_pos = nullptr;

	bool m_biasinuse = true;

#ifdef GPU
	GTYPE  *m_weight_gpu = nullptr;
	GTYPE  *m_b_gpu = nullptr;
	GTYPE  *m_weight_diff_gpu = nullptr;
	GTYPE  *m_b_diff_gpu = nullptr;
#endif
};


#endif