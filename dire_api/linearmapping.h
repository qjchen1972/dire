/*
* Copyright (c) 2017
* All rights reserved.
*
* �ļ�����: linearmapping.h
* �ļ���ʶ:
* ժ    Ҫ: �Ѿ���������һһ����ӳ�䣬���ھ��֮�󣬻����Ӳ��ٲ����������þ����Ľ����ø��߶�����
*
* ��ǰ�汾: 1.0
* ��    ��: chen.qian.jiang
* ��ʼ����: 2018-02-19
* �������:
* ����˵����
* �޸�����        �汾��     �޸���          �޸�����
* -----------------------------------------------
* 2018/02/19      V1.0       ��ǭ��          ����
*/
#pragma once
#ifndef _LINEARMAPPING__H
#define _LINEARMAPPING__H

#include "layer.h"

class LinearMapping : public Clayer
{
public:
	LinearMapping()
	{
		m_layer_type = LINEARMAPPING_LAYER;
	}
	~LinearMapping() {}

	void init_space()
	{
#ifdef GPU
		set_const();
		HANDLE_ERROR(cudaMalloc((void **)&m_bda_gpu, sizeof(GTYPE) * m_fw_batch_size));
		HANDLE_ERROR(cudaMalloc((void **)&m_bda_diff_gpu, sizeof(GTYPE) *m_fw_batch_size));
#endif
	}

	void init()
	{
		init_size();

		m_bda = new GTYPE[m_fw_batch_size];
		m_bda_diff = new GTYPE[m_fw_batch_size];

		if (m_net_status == PARAM_CHECK)
		{
			m_bda_check = new GTYPE[m_fw_batch_size];
		}

		if (g_config.m_study_mode == ADAM_STUDY)
		{
			m_bda_one = new GTYPE[m_fw_batch_size];
			m_bda_two = new GTYPE[m_fw_batch_size];
		}
		if (g_config.m_study_mode == MOMENTUM_STUDY)
		{
			m_bda_diff_before = new GTYPE[m_fw_batch_size];
		}
	}

	void init_param(ifstream *f = nullptr)
	{
		if (m_net_status == PARAM_CHECK)
			sprintf(m_check_file, "%s%d.txt", LINEARMAPPING_CHECK_LOG, m_layer_num);

		if (m_net_status == CONT_TRAINING || m_net_status == TESTING)
		{
			read_param(f);
			return;
		}

		//��ʼ��Ȩֵ
		for (int i = 0; i < m_fw_batch_size; i++)
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
		st.in3 = m_bda_gpu;
		st.start = 0;
		st.len = m_fw_buff_size;
		gpu_cal(st, FN_LINEARMAPPING_FW);
#else		
		unsigned int   i;
		//�ϲ������ڵ� �Լ�row col �� ����һ��
		for (i = 0; i < m_fw_buff_size; i++)
		{
			unsigned int   batch_x = i / m_fw_batch_size;
			unsigned int   batch_y = i % m_fw_batch_size;
			unsigned int   bw_pos = i % m_bw_batch_size;
			m_fw_output[i] = m_fw_total_input[batch_x*m_bw_batch_size+ bw_pos] + m_bda[batch_y];
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
		gpu_cal(st, FN_LINEARMAPPING_BW);
#else
		unsigned int   i,j;
		//�ϲ������ڵ� �Լ�row col �� ����һ��
		for (i = 0; i < m_bw_buff_size; i++)
		{
			unsigned int batch_x = i / m_bw_batch_size;
			unsigned int   batch_y = i % m_bw_batch_size;
			unsigned int  fw_pos = m_fw_batch_size / m_bw_batch_size;
			m_bw_output[i] = 0;
			for(j = 0; j < fw_pos;j++)
				m_bw_output[i] += m_bw_total_input[batch_x * m_fw_batch_size + j*m_bw_batch_size + batch_y];
		}
#endif
		if (g_config.m_free_type == FUNC_FREE)	free_bw_input();
	}

	void update_weight()
	{
		int  i;
		get_diff();
		for (i = 0; i < m_fw_batch_size; i++)
		{
			if (g_config.m_study_mode == ADAM_STUDY)
			{
				adam_study(m_bda_diff[i], &m_bda_one[i], &m_bda_two[i], &m_bda[i]);
			}
			else if (g_config.m_study_mode == MOMENTUM_STUDY)
			{
				momentum_study(m_bda_diff[i], &m_bda[i], &m_bda_diff_before[i]);
			}
			else
			{
				static_study(m_bda_diff[i], &m_bda[i]);
			}
		}
	}

	void create_test_weight(GTYPE(*f)(Clayer*start), Clayer *start)
	{
		GTYPE temp, err1, err2;

		for (int i = 0; i < m_fw_batch_size; i++)
		{	
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
		if (!open_out_file(f, m_check_file)) return;

		int i;
		get_diff();
		f << setprecision(PARAM_PRECISION) << endl;

		for (i = 0; i < m_fw_batch_size; i++)
		{
			if (i % m_fw_col == 0)  f << endl;
			if (i % m_fw_node_size == 0) f << endl;
			f << m_bda_check[i] << " -- " << regular_proc(m_bda_diff[i], m_bda[i]) << ",  ";
		}
		f << endl;
		f << endl;	
		f.close();
	}


private:

	bool write_param(ofstream *f)
	{
		(*f).write((char*)m_bda, sizeof(GTYPE)*m_fw_batch_size);
		return true;
	}

	bool read_param(ifstream *f)
	{
		(*f).read((char*)m_bda, sizeof(GTYPE)*m_fw_batch_size);
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
		HANDLE_ERROR(cudaMemcpy(m_bda_gpu, m_bda, sizeof(GTYPE) *m_fw_batch_size, cudaMemcpyHostToDevice));
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
		if (!m_bw_total_input)
		{
			create_bw_input();
			get_bw_input();
		}	

#ifdef GPU		
		struct gpu_t st;	
		st.result = m_bda_diff_gpu;
		st.in2 = m_bw_total_input;
		st.in2_layer = m_layer_num;
		st.start = 0;
		st.len = m_fw_batch_size;
		st.batchsize = m_batch_num;
		gpu_cal(st, FN_LINEARMAPPING_UPDATE);

		HANDLE_ERROR(cudaMemcpy(m_bda_diff, m_bda_diff_gpu, sizeof(GTYPE) * m_fw_batch_size, cudaMemcpyDeviceToHost));
#else
		int i;
		unsigned int   j;
		for (i = 0; i < m_fw_batch_size; i++)
		{
			m_bda_diff[i] = 0;
			for (j = 0; j < m_batch_num; j++)
			{
				unsigned int   pos = j * m_fw_batch_size + i;
				m_bda_diff[i] += m_bw_total_input[pos];
			}
			m_bda_diff[i] = m_bda_diff[i] / m_batch_num;
		}
#endif
		if (g_config.m_free_type != NERVER_FREE)
		{
			free_bw_input();
			free_next_space();
		}
	}

	GTYPE *m_bda = nullptr; //����bn��ѧϰ����
	GTYPE *m_bda_diff = nullptr;

	GTYPE *m_bda_check = nullptr; //����bn��ѧϰ����У��ֵ

	//adamѧϰ
	GTYPE *m_bda_one = nullptr;
	GTYPE *m_bda_two = nullptr;
	//����ѧϰ
	GTYPE *m_bda_diff_before = nullptr;

#ifdef GPU
	GTYPE *m_bda_gpu = nullptr;
	GTYPE *m_bda_diff_gpu = nullptr;
#endif
};

#endif