/*
* Copyright (c) 2017
* All rights reserved.
*
* 文件名称: conv.h
* 文件标识: 
* 摘    要: 卷积处理，也包含了全连接的处理（可以看成是1*n的卷积核 或者看成1*1) 
*            
*
* 当前版本: 1.0
* 作    者: chen.qian.jiang
* 开始日期: 2017-06-14
* 完成日期:
* 其它说明：
* 修改日期        版本号     修改人          修改内容
* -----------------------------------------------
* 2017/06/24      V1.0       陈黔江          创建
* 2017/07/17      V1.0       陈黔江          增加gpu
*/

#pragma once
#ifndef _CONV__H
#define _CONV__H


#include "layer.h"

class Conv: public Clayer
{
public:
	Conv()
	{
		m_layer_type = CONV_LAYER;
	}
	~Conv(){}
	
	void init_space()
	{
#ifdef GPU
		int kernel_buff_size = m_fw_node_num*m_bw_node_num*m_kernel_row*m_kernel_col;
		set_const();
		HANDLE_ERROR(cudaMalloc((void **)&m_kernel_gpu, sizeof(GTYPE)*kernel_buff_size));
		HANDLE_ERROR(cudaMalloc((void **)&m_kernel_diff_gpu, sizeof(GTYPE)*kernel_buff_size));
		if (m_biasinuse)
		{
			HANDLE_ERROR(cudaMalloc((void **)&m_kernel_b_gpu, sizeof(GTYPE)*m_fw_node_num));
			HANDLE_ERROR(cudaMalloc((void **)&m_kernel_b_diff_gpu, sizeof(GTYPE)*m_fw_node_num));
	     }
#endif
	}

	void init()
	{
		init_size();

		unsigned int   kernel_buff_size = m_fw_node_num * m_bw_node_num*m_kernel_row*m_kernel_col;
		m_kernel = new GTYPE[kernel_buff_size];
		m_kernel_diff = new GTYPE[kernel_buff_size];

		if (m_biasinuse)
		{
			m_kernel_b = new GTYPE[m_fw_node_num];			
			m_kernel_b_diff = new GTYPE[m_fw_node_num];
		}

		if (m_net_status == PARAM_CHECK)
		{
			m_kernel_check = new GTYPE[kernel_buff_size];
			if (m_biasinuse)	m_kernel_b_check = new GTYPE[m_fw_node_num];
		}
		if (g_config.m_study_mode == ADAM_STUDY)
		{
			m_kernel_one = new GTYPE[kernel_buff_size];
			m_kernel_two = new GTYPE[kernel_buff_size];
			if (m_biasinuse)
			{
				m_kernel_b_one = new GTYPE[m_fw_node_num];
				m_kernel_b_two = new GTYPE[m_fw_node_num];
			}
		}
		else if (g_config.m_study_mode == MOMENTUM_STUDY)
		{
			if (m_biasinuse)	m_kernel_b_diff_before = new GTYPE[m_fw_node_num];
			m_kernel_diff_before = new GTYPE[kernel_buff_size];
		}
	}

	void init_param(ifstream *f = nullptr)
	{
		/*sprintf(m_param_file, "%s%d.txt", CONV_PARAM_FILE, m_layer_num);
		ifstream ff;

		char file[FILENAME_LEN];
		sprintf(file, "%s/%s", g_config.m_work_param_dir, m_param_file);
		if (!open_in_file(ff, file)) return ;

		unsigned int   i, m, j, k;
		unsigned int   kernel_node_size = m_bw_node_num*m_kernel_row*m_kernel_col;
		unsigned int   kernel_block_size = m_kernel_row*m_kernel_col;

		for (i = 0; i < m_fw_node_num; i++)
		{				
			ff >> m_kernel_b[i];
			for (m = 0; m < m_bw_node_num; m++)
				for (j = 0; j < m_kernel_row; j++)
					for (k = 0; k < m_kernel_col; k++)
						ff >> m_kernel[i*kernel_node_size + m * kernel_block_size + j *m_kernel_col + k];
		}
		ff.close();
		return;
		*/

		if (m_net_status == PARAM_CHECK)
			sprintf(m_check_file, "%s%d.txt", CONV_CHECK_LOG, m_layer_num);

		if (m_net_status == CONT_TRAINING || m_net_status == TESTING)
		{
			read_param(f);
			return;
		}

		int kernel_buff_size = m_fw_node_num*m_bw_node_num*m_kernel_row*m_kernel_col;
		int kernel_node_size = m_bw_node_num*m_kernel_row*m_kernel_col;

		//给weight赋值
		//为了保证方差的传递保持恒定，若前层是relu的激活层, var(yk) = n * var(w)*var(yk-1)/2 + var(b)。
		//需要让var(w) = 2/n,并且var(b)->0
		// 若是均匀分布，w在(-sqrt(6/n),sqrt(6/n)  b 在(-sqrt(3*e),sqrt(3*e))
		//高斯分布 w-N(0,2/n) b-N(0,e)
		//----------------------------------
		// 若前层是leakyrelu, var(yk) = n*var(w) *(1+a*a)*var(yk-1)/2 + var(b)
		//var(w) = 2/((1+a*a)*n),并且var(b)->0
		//若是均匀分布  w在(-sqrt(6/((1+a*a)*n),sqrt(6/((1+a*a)*n)),b 在(-sqrt(3*e),sqrt(3*e))
		//高斯分布w-N(0,2/((1+a*a)*n)), b-N(0,e)
		//--------------------------------------------
		//若前层是其他，则var(yk) = n*var(w)*var(yk-1) + var(b)
		//需要让var(w) = 1/n,并且var(b)-> 0
		// 若是均匀分布，w在(-sqrt(3/n),sqrt(3/n)  b 在(-sqrt(3*e),sqrt(3*e))
		//高斯分布 w-N(0,1/n) b-N(0,e)	
		GTYPE  t1 = 1.0 / kernel_node_size;
		GTYPE  t3 = sqrt(3.0*t1); //没使用relu的均匀分布
		GTYPE  t4 = sqrt(6.0*t1); //使用了relu的均匀分布
		GTYPE  t5 = t1;     //没使用relu的高斯分布
		GTYPE  t6 = 2 * t1;   //使用了relu的高斯分布

		if (m_biasinuse)
		{
			for (int i = 0; i < m_fw_node_num; i++)
			{
				//bias 只需要方差 -> 0
				if (g_config.m_initweight_mode == XAVIER)
				{
					m_kernel_b[i] = uniform_rand(-sqrt(3 * INFINIT_NUM), sqrt(3 * INFINIT_NUM));
				}
				else if (g_config.m_initweight_mode == MARS)
				{
					m_kernel_b[i] = gauss_rand(0, INFINIT_NUM);
				}
			}
		}

		for (int i = 0; i < kernel_buff_size; i++)
		{
			if (g_config.m_initweight_mode == XAVIER)
			{
				if (!strcmp(m_prev_layer[0]->m_layer_type, ACIVE_LAYER))
				{
					int  type = dynamic_cast<Active*>(m_prev_layer[0])->get_atcive_type();
					if (type == RELU)
					{
						m_kernel[i] = uniform_rand(-t4, t4);
					}
					else if (type == LEAKY_RELU)
					{
						GTYPE  t7 = t1 / (1 + dynamic_cast<Active*>(m_prev_layer[0])->get_leakey_relu()*dynamic_cast<Active*>(m_prev_layer[0])->get_leakey_relu());
						GTYPE  t8 = sqrt(6.0*t7);
						m_kernel[i] = uniform_rand(-t8, t8);
					}
					else
					{
						m_kernel[i] = uniform_rand(-t3, t3);
					}
				}
				else
				{
					m_kernel[i] = uniform_rand(-t3, t3);
				}
			}
			else if (g_config.m_initweight_mode == MARS)
			{
				if (!strcmp(m_prev_layer[0]->m_layer_type, ACIVE_LAYER))
				{
					int  type = dynamic_cast<Active*>(m_prev_layer[0])->get_atcive_type();
					if (type == RELU)
					{
						m_kernel[i] = gauss_rand(0, t6);
					}
					else if (type == LEAKY_RELU)
					{
						GTYPE  t7 = t1 / (1 + dynamic_cast<Active*>(m_prev_layer[0])->get_leakey_relu()*dynamic_cast<Active*>(m_prev_layer[0])->get_leakey_relu());
						GTYPE  t8 = 2.0*t7;
						m_kernel[i] = gauss_rand(0, t8);
					}
					else
					{
						m_kernel[i] = gauss_rand(0, t5);
					}
				}
				else
				{
					m_kernel[i] = gauss_rand(0, t5);
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
		st.in1 = m_kernel_gpu;
		if(m_biasinuse)	st.in2 = m_kernel_b_gpu;
		st.in3 = m_fw_total_input;
		st.start = 0;
		st.len = m_fw_buff_size;
		gpu_cal(st, FN_CONV_FW);
#else

		int kernel_block_size = m_kernel_row*m_kernel_col;
		int kernel_node_size = m_bw_node_num*kernel_block_size;

		unsigned int   i,j;		
		for (i = 0; i < m_fw_buff_size; i++)
		{			
			unsigned int   batch_x = i / m_fw_batch_size;
			unsigned int   batch_y = i % m_fw_batch_size;

			unsigned int   node_x = batch_y / m_fw_node_size;
			unsigned int   node_y  = batch_y % m_fw_node_size;

			int x = node_y / m_fw_col;
			int y = node_y % m_fw_col;

			if (x < m_top_padding || x >= m_fw_row - m_bottom_padding || y < m_left_padding || y >= m_fw_col - m_right_padding)
			{
				m_fw_output[i] = 0;
				continue;
			}

			int s = (x - m_top_padding)*m_row_stride;
			int t = (y - m_left_padding)*m_col_stride;

			GTYPE temp = 0;
			if(m_biasinuse) temp = m_kernel_b[node_x];

			//kernel 的row pos
			unsigned int    ker_row_pos = node_x*kernel_node_size;
			//input在哪个batch
			unsigned int    input_batch_pos = batch_x*m_bw_batch_size;

			for (j = 0; j < kernel_node_size; j++)
			{
				unsigned int   prev_node_x = j / kernel_block_size;
				unsigned int   prev_node_y = j % kernel_block_size;

				int u = prev_node_y / m_kernel_col;
				int v = prev_node_y % m_kernel_col;

				if (s + u < m_bw_row && t + v < m_bw_col)								
					temp += m_kernel[ker_row_pos + j] *
						m_fw_total_input[input_batch_pos + prev_node_x*m_bw_node_size + (s + u)*m_bw_col + t + v];
			}
			m_fw_output[i] = temp;
		}
#endif
		if (g_config.m_free_type == FUNC_FREE)	free_fw_input();
	}

	void backword_proc()
	{
		if (m_next_num <= 0 ) return;
		create_bw_input();
		get_bw_input();
		create_bw_output();
		
#ifdef GPU
		struct gpu_t st;
		st.result = m_bw_output;
		st.result_layer = m_layer_num;
		st.in1 = m_kernel_gpu;
		st.in2 = m_bw_total_input;
		st.start = 0;
		st.len = m_bw_buff_size;
		gpu_cal(st, FN_CONV_BW);
#else

		unsigned int   kernel_block_size = m_kernel_row*m_kernel_col;
		unsigned int   kernel_noderow_size = m_fw_node_num*kernel_block_size;
		unsigned int   i,j;
		unsigned int   kernel_nodecol_size = m_bw_node_num*kernel_block_size;

		for (i = 0; i < m_bw_buff_size; i++)
		{
			unsigned int   batch_x = i / m_bw_batch_size;
			unsigned int   batch_y = i % m_bw_batch_size;

			unsigned int   prev_node_x = batch_y / m_bw_node_size;
			unsigned int   prev_node_y = batch_y % m_bw_node_size;

			int x = prev_node_y / m_bw_col;
			int y = prev_node_y % m_bw_col;

			GTYPE temp = 0;
			//kernel 位置
			unsigned int   ker_pos =  prev_node_x*kernel_block_size;
			// bw input 所在的batch
			unsigned int   bw_batch_pos = batch_x*m_fw_batch_size;
			for (j = 0; j < kernel_noderow_size; j++)
			{
				unsigned int   node_x = j / kernel_block_size;
				unsigned int   node_y = j % kernel_block_size;

				int u = node_y / m_kernel_col;
				int v = node_y % m_kernel_col;

				int s = (x - u) / m_row_stride + m_top_padding;
				int t = (y - v) / m_col_stride + m_left_padding;
				
				if (x - u >= 0 && y - v >= 0 && (x - u) % m_row_stride == 0 && (y - v) % m_col_stride == 0 &&
					(x - u) / m_row_stride < m_fw_real_row && (y - v) / m_col_stride < m_fw_real_col)
				{
					temp += m_kernel[node_x*kernel_nodecol_size + ker_pos + node_y] * m_bw_total_input[bw_batch_pos + node_x*m_fw_node_size + s*m_fw_col + t];			
				}
			}
			m_bw_output[i] = temp;			
		}
#endif
		if (g_config.m_free_type == FUNC_FREE)	free_bw_input();
	}

	void update_weight()
	{		
		unsigned int   i;	
		get_diff();
		unsigned int   kernel_buff_size = m_fw_node_num*m_bw_node_num*m_kernel_row*m_kernel_col;

		for (i = 0; i < kernel_buff_size; i++)
		{
			if (g_config.m_study_mode == ADAM_STUDY)
				adam_study(m_kernel_diff[i], &m_kernel_one[i], &m_kernel_two[i], &m_kernel[i]);
			else if (g_config.m_study_mode == MOMENTUM_STUDY)
				momentum_study(m_kernel_diff[i], &m_kernel[i], &m_kernel_diff_before[i]);
			else
				static_study(m_kernel_diff[i], &m_kernel[i]);
		}
		if (m_biasinuse)
		{
			for (i = 0; i < m_fw_node_num; i++)
			{
				if (g_config.m_study_mode == ADAM_STUDY)
					adam_study(m_kernel_b_diff[i], &m_kernel_b_one[i], &m_kernel_b_two[i], &m_kernel_b[i]);
				else if (g_config.m_study_mode == MOMENTUM_STUDY)
					momentum_study(m_kernel_b_diff[i], &m_kernel_b[i], &m_kernel_b_diff_before[i]);
				else
					static_study(m_kernel_b_diff[i], &m_kernel_b[i]);
			}
		}
	}

	// 利用导数的定义 ：df / dx = lim(f(x+e) - f(x-e))/ 2e e->0
	//来校验导数计算是否正确
	void create_test_weight(GTYPE(*f)(Clayer *start), Clayer *start)
	{
		GTYPE temp;
		GTYPE err1, err2;
		unsigned int   i;

		unsigned int   kernel_buff_size = m_fw_node_num*m_bw_node_num*m_kernel_row*m_kernel_col;

		if (m_biasinuse)
		{
			for (i = 0; i < m_fw_node_num; i++)
			{
				temp = m_kernel_b[i];
				m_kernel_b[i] = temp - INFINIT_NUM;
				err1 = f(start);
				m_kernel_b[i] = temp + INFINIT_NUM;
				err2 = f(start);


				m_kernel_b_check[i] = (err2 - err1 + regular_check(temp)) / (2.0 * INFINIT_NUM);
				m_kernel_b[i] = temp;
			}
		}

		for (i = 0; i < kernel_buff_size; i++)
		{
			temp = m_kernel[i];

			m_kernel[i] = temp - INFINIT_NUM;
			err1 = f(start);
			m_kernel[i] = temp + INFINIT_NUM;
			err2 = f(start);
			m_kernel_check[i] = (err2 - err1 + regular_check(temp)) / (2.0 * INFINIT_NUM);
			m_kernel[i] = temp;
		}
	}

	//只用于校验导数
	//和update_weight一样
	void check_weight()
	{		
		unsigned int   i;
		ofstream f;		

		if (!open_out_file(f, m_check_file))	return;				

		get_diff();

		unsigned int   kernel_buff_size = m_fw_node_num*m_bw_node_num*m_kernel_row*m_kernel_col;
		unsigned int   kernel_node_size = m_bw_node_num*m_kernel_row*m_kernel_col;
		unsigned int   kernel_block_size = m_kernel_row*m_kernel_col;
	
		f << setprecision(PARAM_PRECISION)<<endl;
		for (i = 0; i < kernel_buff_size; i++)
		{
			if (i % m_kernel_col == 0)  f << endl;
			if (i % kernel_block_size == 0) f << endl;
			if (i % kernel_node_size == 0) f << endl;
			f << m_kernel_check[i] << " -- " << regular_proc(m_kernel_diff[i], m_kernel[i]) << "  ";
		}
		f << endl;
		f << endl;

		if (m_biasinuse)
		{
			for (i = 0; i < m_fw_node_num; i++)
			{
				f << "kernel_b[" << i << "] " << m_kernel_b_check[i] << " -- " << regular_proc(m_kernel_b_diff[i], m_kernel_b[i]) << endl;
			}
		}
		f.close();
	}


	//设置卷积核大小以及stride.
	void set_kernel_param(int row, int col, int row_stride, int col_stride)
	{
		m_kernel_row = row;
		m_kernel_col = col;
		m_row_stride = row_stride;
		m_col_stride = col_stride;
	}

	void close_bias() { m_biasinuse = false; }

private:	

	bool write_param(ofstream *f)
	{
		if (m_biasinuse) (*f).write((char*)m_kernel_b, sizeof(GTYPE)*m_fw_node_num);
		(*f).write((char*)m_kernel, sizeof(GTYPE)* m_fw_node_num * m_bw_node_num*m_kernel_row*m_kernel_col);
		return true;
	}

	bool read_param(ifstream *f)
	{
		if (m_biasinuse) (*f).read((char*)m_kernel_b, sizeof(GTYPE)*m_fw_node_num);
		if ((*f).eof()) return false;
		(*f).read((char*)m_kernel, sizeof(GTYPE)*m_fw_node_num * m_bw_node_num*m_kernel_row*m_kernel_col);
		if ((*f).eof()) return false;
		return true;
	}

	void  set_param()
	{
#ifdef GPU
		int kernel_buff_size = m_fw_node_num*m_bw_node_num*m_kernel_row*m_kernel_col;
		if (m_biasinuse)	HANDLE_ERROR(cudaMemcpy(m_kernel_b_gpu, m_kernel_b, sizeof(GTYPE) *m_fw_node_num, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(m_kernel_gpu, m_kernel, sizeof(GTYPE) *kernel_buff_size, cudaMemcpyHostToDevice));
#endif
	}

#ifdef GPU
	void set_const()
	{
		g_cpu_layer[m_layer_num].type.conv.top_padding = m_top_padding;
		g_cpu_layer[m_layer_num].type.conv.bottom_padding = m_bottom_padding;
		g_cpu_layer[m_layer_num].type.conv.left_padding = m_left_padding;
		g_cpu_layer[m_layer_num].type.conv.right_padding = m_right_padding;
		g_cpu_layer[m_layer_num].type.conv.row_stride = m_row_stride;
		g_cpu_layer[m_layer_num].type.conv.col_stride = m_col_stride;
		g_cpu_layer[m_layer_num].type.conv.kernel_row = m_kernel_row;
		g_cpu_layer[m_layer_num].type.conv.kernel_col = m_kernel_col;
		Clayer::set_const();
	}
#endif

	void init_size()
	{
		
		init_base_size();

		//若是没有设置卷积核的大小，应该是和前一层输入矩阵一样大小
		if (m_kernel_row == 0)
		{
			m_kernel_row = m_bw_row;
			m_kernel_col = m_bw_col;
		}

		//计算fw实际的大小	
		m_fw_real_row = (m_bw_row - m_kernel_row) / m_row_stride + 1;
		m_fw_real_col = (m_bw_col - m_kernel_col) / m_col_stride + 1;
		if ((m_bw_row - m_kernel_row) % m_row_stride != 0)
		{
			if((m_bw_row - m_kernel_row) % m_row_stride + m_kernel_row  > m_row_stride)
				m_fw_real_row += 1;
		}
		if ((m_bw_col - m_kernel_col) % m_col_stride != 0)
		{
			if ((m_bw_col - m_kernel_col) % m_col_stride + m_kernel_col  > m_col_stride)
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
		unsigned int   kernel_block_size = m_kernel_row*m_kernel_col;
		unsigned int   kernel_buff_size = m_fw_node_num*m_bw_node_num*kernel_block_size;
		unsigned int   nodes_len = m_batch_num *m_fw_row*m_fw_col;		

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
		st.result = m_kernel_diff_gpu;
		st.result_layer = m_layer_num;
		st.in1 = m_fw_total_input;
		st.in2 = m_bw_total_input;
		st.start = 0;
		st.len = kernel_buff_size;
		st.batchsize = m_batch_num;
		gpu_cal(st, FN_CONV_UPDATE);
		

		/*struct gpu_t st;
		st.result = m_fw_total_input;
		st.result_layer = m_layer_num;
		st.in1 = m_bw_total_input;
		st.start = 0;
		st.len = nodes_len;
		st.sum_result = m_kernel_diff_gpu;
		st.sum_result_count= kernel_buff_size;
		gpu_sum(st, FN_CONV_UPDATE);		
		*/

		if (m_biasinuse)
		{
			st.result = m_bw_total_input;
			st.result_layer = m_layer_num;
			st.start = 0;
			st.len = nodes_len;
			st.sum_result = m_kernel_b_diff_gpu;
			st.sum_result_count = m_fw_node_num;
			st.batchsize = m_batch_num;
			gpu_sum(st, FN_CONV_UPDATE_SUM);
		}
	
		if (m_biasinuse)
			HANDLE_ERROR(cudaMemcpy(m_kernel_b_diff, m_kernel_b_diff_gpu, sizeof(GTYPE) *m_fw_node_num, cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(m_kernel_diff, m_kernel_diff_gpu, sizeof(GTYPE) * kernel_buff_size, cudaMemcpyDeviceToHost));

#else		
		unsigned int   i, j;
		int kernel_node_size = m_bw_node_num*kernel_block_size;
		for (i = 0; i < kernel_buff_size; i++)
		{
			unsigned int   node_x = i / kernel_node_size;
			unsigned int   node_y = i %  kernel_node_size;


			unsigned int   p_node_x = node_y / kernel_block_size;
			unsigned int   p_node_y = node_y %  kernel_block_size;


			int x = p_node_y / m_kernel_col;
			int y = p_node_y % m_kernel_col;

			m_kernel_diff[i] = 0;
			unsigned int   pre_node_pos = p_node_x*m_bw_node_size;
			unsigned int   node_pos = node_x*m_fw_node_size;

			for (j = 0; j < nodes_len; j++)
			{
				unsigned int   batch_x = j / m_fw_node_size;
				unsigned int   batch_y = j % m_fw_node_size;

				int u = batch_y / m_fw_col;
				int v = batch_y % m_fw_col;

				if (u < m_top_padding || v < m_left_padding || u >= m_fw_row - m_bottom_padding || v >= m_fw_col - m_right_padding
					|| (u - m_top_padding) * m_row_stride + x >= m_bw_row || (v - m_left_padding) * m_col_stride + y >= m_bw_col)
					continue;

				m_kernel_diff[i] += m_fw_total_input[batch_x*m_bw_batch_size + pre_node_pos +
					((u - m_top_padding) * m_row_stride + x)*m_bw_col + ((v - m_left_padding) * m_col_stride + y)] * m_bw_total_input[batch_x*m_fw_batch_size + node_pos + u * m_fw_col + v];
			}
			m_kernel_diff[i] = m_kernel_diff[i] / m_batch_num;
		}

		if (m_biasinuse)
		{
			for (i = 0; i < m_fw_node_num; i++)
			{
				m_kernel_b_diff[i] = 0;
				for (j = 0; j < nodes_len; j++)
				{
					unsigned int   batch_x = j / m_fw_node_size;
					unsigned int   batch_y = j % m_fw_node_size;

					int u = batch_y / m_fw_col;
					int v = batch_y % m_fw_col;

					if (u < m_top_padding || v < m_left_padding || u >= m_fw_row - m_bottom_padding || v >= m_fw_col - m_right_padding)
						continue;
					m_kernel_b_diff[i] += m_bw_total_input[batch_x * m_fw_batch_size + i * m_fw_node_size + batch_y];
				}
				m_kernel_b_diff[i] = m_kernel_b_diff[i] / m_batch_num;
			}
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
	

	GTYPE *m_kernel = nullptr;//卷积核
	GTYPE *m_kernel_b = nullptr;//卷积核的bias

	//梯度
	GTYPE *m_kernel_diff = nullptr;
	GTYPE *m_kernel_b_diff = nullptr;

	GTYPE *m_kernel_check = nullptr;//卷积核参数校验时保存的检验值	
	GTYPE *m_kernel_b_check = nullptr; //卷积核bias参数校验时保存的检验值

	
	//用于Adam学习
	GTYPE *m_kernel_one = nullptr;//一阶矩
	GTYPE *m_kernel_b_one = nullptr; //

	GTYPE *m_kernel_two = nullptr;//二阶矩
	GTYPE *m_kernel_b_two = nullptr; //

	//用于冲量学习 
	GTYPE *m_kernel_diff_before = nullptr;
	GTYPE *m_kernel_b_diff_before = nullptr; 

	int  m_top_padding = 0;
	int  m_bottom_padding = 0;
	int  m_left_padding = 0;
	int  m_right_padding = 0;

	int  m_kernel_row = 0; //卷积核的行
	int  m_kernel_col = 0; //卷积核的列	
	int  m_row_stride = 1;
	int  m_col_stride = 1;		
	bool m_biasinuse = true;

#ifdef GPU
	GTYPE *m_kernel_gpu = nullptr;//卷积核
	GTYPE *m_kernel_b_gpu = nullptr;//卷积核的bias
	GTYPE *m_kernel_diff_gpu = nullptr; //卷积核的梯度
	GTYPE *m_kernel_b_diff_gpu = nullptr;
#endif
};

#endif
