/*
* Copyright (c) 2017
* All rights reserved.
*
* �ļ�����: scale.h
* �ļ���ʶ:
* ժ    Ҫ: ��һ��������Ҫ����ΪBN��ᵼ��bias�����������ã����Լ���һ��ƽ�Ʋ�(scale)��֮����û�в���bn���ԭ����
*           Ϊ���ó�����Ʊ�ü򵥣�����һ���������������Ǻϲ������ò�������ø��ӣ������޸ĺ�ά�������㡣
*            ������caffҲ��û�в��룬��û���ף��Լ����������ף�����Ӧ�ö������ԭ��
*
*
* ��ǰ�汾: 1.0
* ��    ��: chen.qian.jiang
* ��ʼ����: 2017-06-29
* �������:
* ����˵����
* �޸�����        �汾��     �޸���          �޸�����
* -----------------------------------------------
* 2017/06/29      V1.0       ��ǭ��          ����
* 2017/07/17      V1.0       ��ǭ��          ����gpu
*/
#pragma once
#ifndef _SCALE__H
#define _SCALE__H

#include "layer.h"

class Scale : public Clayer
{
public:
	Scale() 
	{
		m_layer_type = SCALE_LAYER;	
	}
	~Scale() {}

	void init_space()
	{
#ifdef GPU
		set_const();
		HANDLE_ERROR(cudaMalloc((void **)&m_gem_gpu, sizeof(GTYPE) *m_fw_node_num));
		HANDLE_ERROR(cudaMalloc((void **)&m_gem_diff_gpu, sizeof(GTYPE)*m_fw_node_num));
		HANDLE_ERROR(cudaMalloc((void **)&m_bda_gpu, sizeof(GTYPE) *m_fw_node_num));
		HANDLE_ERROR(cudaMalloc((void **)&m_bda_diff_gpu, sizeof(GTYPE) *m_fw_node_num));
#endif
	}

	void init()
	{		
		init_size();

		m_gem = new GTYPE[m_fw_node_num];
		m_bda = new GTYPE[m_fw_node_num];
		m_gem_diff = new GTYPE[m_fw_node_num];
		m_bda_diff = new GTYPE[m_fw_node_num];

		if (m_net_status == PARAM_CHECK)
		{
			m_gem_check = new GTYPE[m_fw_node_num];
			m_bda_check = new GTYPE[m_fw_node_num];
		}

		if (g_config.m_study_mode == ADAM_STUDY)
		{
			m_gem_one = new GTYPE[m_fw_node_num];
			m_bda_one = new GTYPE[m_fw_node_num];
			m_gem_two = new GTYPE[m_fw_node_num];
			m_bda_two = new GTYPE[m_fw_node_num];
		}
		if (g_config.m_study_mode == MOMENTUM_STUDY)
		{
			m_gem_diff_before = new GTYPE[m_fw_node_num];
			m_bda_diff_before = new GTYPE[m_fw_node_num];
		}
	}


	void init_param(ifstream *f = nullptr)
	{
		/*sprintf(m_param_file, "%s%d.txt", SCALE_PARAM_FILE, m_layer_num);
		ifstream ff;
		char file[FILENAME_LEN];
		sprintf(file, "%s/%s", g_config.m_work_param_dir, m_param_file);
		if (!open_in_file(ff, file)) return;
		for (int i = 0; i < m_fw_node_num; i++)
		{
			ff >> m_gem[i] >> m_bda[i];
		}
		ff.close();
		return ;
		*/

		if (m_net_status == PARAM_CHECK)
			sprintf(m_check_file, "%s%d.txt", SCALE_CHECK_LOG, m_layer_num);

		if (m_net_status == CONT_TRAINING || m_net_status == TESTING)
		{
			read_param(f);
			return;
		}

		//��ʼ��Ȩֵ
		for (int i = 0; i < m_fw_node_num; i++)
		{
			//bda ֻ��Ҫ���� -> 0
			if (g_config.m_initweight_mode == XAVIER)
			{
				m_bda[i] = uniform_rand(-sqrt(3 * INFINIT_NUM), sqrt(3 * INFINIT_NUM));
			}
			else if (g_config.m_initweight_mode == MARS)
			{
				m_bda[i] = gauss_rand(0, INFINIT_NUM);
			}
			//weight 
			//����ǰ����relu��var(yk) = 1/2*var(w)var(yk-1) + var(b) 
			// var(w) = 2
			//����leakyrelu, var(yk) = 1/2 *(1+a*a) * var(yk-1) * var(w) + var(b)
			// var(w) = 2/(1+a*a)
			//����������var(yk) = var(w)var(yk-1) + var(b)
			// var(w) = 1
			if (g_config.m_initweight_mode == XAVIER)
			{
				if (!strcmp(m_prev_layer[0]->m_layer_type, ACIVE_LAYER))
				{
					int  type = dynamic_cast<Active*>(m_prev_layer[0])->get_atcive_type();
					if (type == RELU)
					{
						m_gem[i] = uniform_rand(-sqrt(6), sqrt(6));
					}
					else if (type == LEAKY_RELU)
					{
						GTYPE  t7 = 1.0 / (1 + dynamic_cast<Active*>(m_prev_layer[0])->get_leakey_relu()*dynamic_cast<Active*>(m_prev_layer[0])->get_leakey_relu());
						GTYPE  t8 = sqrt(6.0*t7);
						m_gem[i] = uniform_rand(-t8, t8);
					}
					else
					{
						m_gem[i] = uniform_rand(-sqrt(3), sqrt(3));
					}
				}
				else
				{
					m_gem[i] = uniform_rand(-sqrt(3), sqrt(3));
				}
			}
			else if (g_config.m_initweight_mode == MARS)
			{
				if (!strcmp(m_prev_layer[0]->m_layer_type, ACIVE_LAYER))
				{
					int  type = dynamic_cast<Active*>(m_prev_layer[0])->get_atcive_type();
					if (type == RELU)
					{
						m_gem[i] = gauss_rand(0, 2);
					}
					else if (type == LEAKY_RELU)
					{
						GTYPE  t7 = 1 / (1 + dynamic_cast<Active*>(m_prev_layer[0])->get_leakey_relu()*dynamic_cast<Active*>(m_prev_layer[0])->get_leakey_relu());
						GTYPE  t8 = 2.0*t7;
						m_gem[i] = gauss_rand(0, t8);
					}
					else
					{
						m_gem[i] = gauss_rand(0, 1);
					}
				}
				else
				{
					m_gem[i] = gauss_rand(0, 1);
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
		st.in2 = m_gem_gpu;
		st.in3 = m_bda_gpu;		
		st.start = 0;
		st.len = m_fw_buff_size;
		gpu_cal(st, FN_SCALE_FW);
#else		
		unsigned int   i;
		//�ϲ������ڵ� �Լ�row col �� ����һ��
		for (i = 0; i < m_fw_buff_size; i++)
		{			
			unsigned int   batch_y = i % m_fw_batch_size;
			unsigned int   node_x = batch_y / m_fw_node_size;
			m_fw_output[i] = m_gem[node_x] * m_fw_total_input[i] + m_bda[node_x];
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
		st.in2 = m_gem_gpu;
		st.start = 0;
		st.len = m_bw_buff_size;
		gpu_cal(st, FN_SCALE_BW);
#else
		unsigned int   i;
		//�ϲ������ڵ� �Լ�row col �� ����һ��
		for (i = 0; i < m_bw_buff_size; i++)
		{
			unsigned int   batch_y = i % m_bw_batch_size;
			unsigned int   node_x = batch_y / m_bw_node_size;
			m_bw_output[i] = m_gem[node_x] * m_bw_total_input[i];
		}
#endif
		if (g_config.m_free_type == FUNC_FREE)	free_bw_input();
	}

	void update_weight()
	{
		int  i;
		get_diff();		
		for (i = 0; i < m_fw_node_num; i++)
		{
			if (g_config.m_study_mode == ADAM_STUDY)
			{
				adam_study(m_gem_diff[i], &m_gem_one[i], &m_gem_two[i], &m_gem[i]);
				adam_study(m_bda_diff[i], &m_bda_one[i], &m_bda_two[i], &m_bda[i]);
			}
			else if (g_config.m_study_mode == MOMENTUM_STUDY)
			{
				momentum_study(m_gem_diff[i], &m_gem[i], &m_gem_diff_before[i]);
				momentum_study(m_bda_diff[i], &m_bda[i], &m_bda_diff_before[i]);
			}
			else
			{
				static_study(m_gem_diff[i], &m_gem[i]);
				static_study(m_bda_diff[i], &m_bda[i]);
			}
		}
	}

	void create_test_weight(GTYPE(*f)(Clayer*start), Clayer *start)
	{
		GTYPE temp, err1, err2;
		
		for (int i = 0; i < m_fw_node_num; i++)
		{
			temp = m_gem[i];
			m_gem[i] = temp - INFINIT_NUM;
			err1 = f(start);
			m_gem[i] = temp + INFINIT_NUM;
			err2 = f(start);
			m_gem_check[i] = (err2 - err1 + regular_check(temp)) / (2.0 * INFINIT_NUM);
			m_gem[i] = temp;

			temp = m_bda[i];
			m_bda[i] = temp - INFINIT_NUM;
			err1 = f(start);
			m_bda[i] = temp + INFINIT_NUM;
			err2 = f(start);
			m_bda_check[i] = (err2 - err1 + regular_check(temp)) / (2.0 * INFINIT_NUM);
			m_bda[i] = temp;
		}
	}

	void check_weight()
	{		
		ofstream f;
		if (!open_out_file(f, m_check_file)) return ;		

		int i;
		get_diff();
		f << setprecision(PARAM_PRECISION) << endl;

		for (i = 0; i < m_fw_node_num; i++)
		{			
			f << " gem_check[" << i << "] " << m_gem_check[i] << " " << regular_proc(m_gem_diff[i], m_gem[i]);
			f << "   bda_check[" << i << "]  " << m_bda_check[i] << " " << regular_proc(m_bda_diff[i], m_bda[i]) << endl;
		}
		f.close();		
	}

private:

	bool write_param(ofstream *f)
	{
		(*f).write((char*)m_gem, sizeof(GTYPE)*m_fw_node_num);
		(*f).write((char*)m_bda, sizeof(GTYPE)* m_fw_node_num);
		return true;
	}

	bool read_param(ifstream *f)
	{
		(*f).read((char*)m_gem, sizeof(GTYPE)*m_fw_node_num);
		if ((*f).eof()) return false;
		(*f).read((char*)m_bda, sizeof(GTYPE)* m_fw_node_num);
		if ((*f).eof()) return false;
		return true;
	}
	
#ifdef GPU
	void set_const()
	{
		Clayer::set_const();
	}
#endif


	void  set_param()
	{
#ifdef GPU
		HANDLE_ERROR(cudaMemcpy(m_gem_gpu, m_gem, sizeof(GTYPE) *m_fw_node_num, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(m_bda_gpu, m_bda, sizeof(GTYPE) *m_fw_node_num, cudaMemcpyHostToDevice));
#endif
	}

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
		st.result = m_bw_total_input;
		st.result_layer = m_layer_num;
		st.start = 0;
		st.len = nodes_len;
		st.sum_result = m_bda_diff_gpu;
		st.sum_result_count = m_fw_node_num;
		st.batchsize = m_batch_num;
		gpu_sum(st, FN_SCALE_UPDATE_BDA_SUM);		

		st.result = m_bw_total_input;
		st.result_layer = m_layer_num;
		st.in1 = m_fw_total_input;
		st.start = 0;
		st.len = nodes_len;
		st.sum_result = m_gem_diff_gpu;
		st.sum_result_count = m_fw_node_num;
		st.batchsize = m_batch_num;
		gpu_sum(st, FN_SCALE_UPDATE_GEM_SUM);		

		HANDLE_ERROR(cudaMemcpy(m_gem_diff, m_gem_diff_gpu, sizeof(GTYPE) *m_fw_node_num, cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(m_bda_diff, m_bda_diff_gpu, sizeof(GTYPE) * m_fw_node_num, cudaMemcpyDeviceToHost));
#else
		int i;
		unsigned int   j;
		for (i = 0; i < m_fw_node_num; i++)
		{
			m_gem_diff[i] = 0;
			m_bda_diff[i] = 0;
			for (j = 0; j < nodes_len; j++)
			{
				unsigned int   batch_x = j / m_fw_node_size;
				unsigned int   batch_y = j % m_fw_node_size;
				unsigned int   pos = batch_x * m_fw_batch_size + i*m_fw_node_size + batch_y;

				m_gem_diff[i] += m_bw_total_input[pos] * m_fw_total_input[pos];
				m_bda_diff[i] += m_bw_total_input[pos];
			}
			m_gem_diff[i] = m_gem_diff[i] / m_batch_num;
			m_bda_diff[i] = m_bda_diff[i] / m_batch_num;
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

	GTYPE *m_gem = nullptr; //����bn��ѧϰ����
	GTYPE *m_bda = nullptr; //����bn��ѧϰ����
	GTYPE *m_gem_diff = nullptr;
	GTYPE *m_bda_diff = nullptr;

	GTYPE *m_gem_check = nullptr; //����bn��ѧϰ����У��ֵ
	GTYPE *m_bda_check = nullptr; //����bn��ѧϰ����У��ֵ
	
	//adamѧϰ
	GTYPE *m_gem_one = nullptr;
	GTYPE *m_bda_one = nullptr;
	GTYPE *m_gem_two = nullptr;
	GTYPE *m_bda_two = nullptr;
	//����ѧϰ
	GTYPE *m_gem_diff_before = nullptr;
	GTYPE *m_bda_diff_before = nullptr;


#ifdef GPU
	GTYPE *m_gem_gpu = nullptr;
	GTYPE *m_bda_gpu = nullptr;
	GTYPE *m_gem_diff_gpu = nullptr;
	GTYPE *m_bda_diff_gpu = nullptr;
#endif
};

#endif