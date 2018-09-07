/*
* Copyright (c) 2017
* All rights reserved.
*
* �ļ�����: layer.h
* �ļ���ʶ:
* ժ    Ҫ: ������
*
*
* ��ǰ�汾: 1.0
* ��    ��: chen.qian.jiang
* ��ʼ����: 2017-06-14
* �������:
* ����˵����
* �޸�����        �汾��     �޸���          �޸�����
* -----------------------------------------------
* 2017/06/14      V1.0       ��ǭ��          ����
* 2017/07/28      V1.0       ��ǭ��          ����gpu
*/

#pragma once
#ifndef _LAYER__H
#define _LAYER__H

#include "mathtool.h"
#include "gpu.h"

class Clayer :public Mathtool
#ifdef GPU
	, public Gpu
#endif
{
public:
	Clayer()
	{
		m_layer_type = GLOBAL_LAYER;
	}

	~Clayer() {}

	//ȫ�ֱ���
	static unsigned int  m_train_num;  //ѵ���˵Ĵ���
	static unsigned int  m_batch_num; //����������Ŀ	
	static GTYPE m_mincost; //��¼ѵ����������Сֵ
	static GTYPE m_best_answer; //������ý��
	static unsigned int m_total_train; //��¼�ܹ���ѵ������
	static int m_net_status; //����״̬
	//ȫ������
	void set_batch_size(unsigned int size) { m_batch_num = size; }
	void set_train_num(unsigned int num) { m_train_num = num; }
	void set_net_status(int status) { m_net_status = status; }
	
	//��ȫ������
	void set_prev_layer(Clayer *layer)
	{
		m_prev_layer[m_prev_num] = layer;
		m_prev_num++;
	}
	void set_next_layer(Clayer *layer)
	{
		m_next_layer[m_next_num] = layer;
		m_next_num++;
	}
	
	//���øò�����λ��
	void set_layer_num(int num) { m_layer_num = num; }
	//
	void set_node_num(int num) { m_fw_node_num = num; }     //��������Ϊ0��ȡ��һ���ֵ
	void set_fw_out(int row, int col) //��������Ϊ0��ȡʵ�ʻ�õĴ�С
	{
		m_fw_row = row;
		m_fw_col = col;
	}

	//�����3��ǰ��
	//2018.1.1 ����4����
	//2018.2.13 �޸�ΪMAX_CONN_NUM = 128
	Clayer *m_prev_layer[MAX_CONN_NUM] = { nullptr };
	Clayer *m_next_layer[MAX_CONN_NUM] = { nullptr };
	int m_prev_num = 0;
	int m_next_num = 0;	
	int  m_fw_del_request = 0; //���fwɾ������==m_next_num����Ҫɾ��ǰ������ڴ�
	int  m_bw_del_request = 0; //���bwɾ������==m_prev_num����Ҫɾ����������ڴ�
	int  m_next_pos[MAX_CONN_NUM] = { 0 }; //���ڸú���������λ��

	//ǰ�����
	GTYPE  *m_fw_output = nullptr;
	//�������
	GTYPE  *m_bw_output = nullptr;
	//ǰ������
	GTYPE  *m_fw_total_input = nullptr;
	//��������
	GTYPE  *m_bw_total_input = nullptr;

	int  m_fw_row = 0; //ǰ���������
	int  m_fw_col = 0; //ǰ���������
	int  m_fw_real_row = 0; //ǰ��ʵ���������
	int  m_fw_real_col = 0;	//ǰ��ʵ���������
	unsigned int  m_fw_node_num = 0; //ǰ������Ľڵ���
	unsigned int    m_fw_buff_size = 0; // ����buf�ĳߴ�
	unsigned int    m_fw_batch_size = 0; //����batch�ĳߴ�
	unsigned int    m_fw_node_size = 0;  //����node�ĳߴ�	


	int  m_bw_row = 0; //�����������
	int  m_bw_col = 0; //�����������
	unsigned int  m_bw_node_num = 0; //��������Ľڵ���
	unsigned int   m_bw_buff_size = 0; // ����buf�ĳߴ�
	unsigned int   m_bw_batch_size = 0; //����batch�ĳߴ�
	unsigned int   m_bw_node_size = 0;  //����node�ĳߴ�	


	unsigned int  m_layer_num = 0;//��ǰ����
	const char* m_layer_type;//��ǰ������͡�

#define FILE_SIZE 128
	char m_param_file[FILE_SIZE]; //�����ļ�
	char m_check_file[FILE_SIZE]; //�ݶ�У���ļ�

	//���ڱ���
	bool is_proc_over = false;
	bool is_check_over = false;
	bool fw_need_write_file = false;

	
	void reset_size()
	{
		m_fw_buff_size = m_batch_num * m_fw_node_num * m_fw_row * m_fw_col;
		m_fw_batch_size = m_fw_node_num * m_fw_row * m_fw_col;
		m_fw_node_size = m_fw_row * m_fw_col;
		m_bw_buff_size = m_batch_num * m_bw_node_num * m_bw_row * m_bw_col;
		m_bw_batch_size = m_bw_node_num * m_bw_row * m_bw_col;
		m_bw_node_size = m_bw_row * m_bw_col;		

		m_fw_del_request = 0;
		m_bw_del_request = 0; 
	}	


	void create_fw_input()
	{
#ifdef GPU		
		if (!m_fw_total_input)
			HANDLE_ERROR(cudaMalloc((void **)&m_fw_total_input, sizeof(GTYPE)*m_bw_buff_size));
     	
#else
			if (!m_fw_total_input)
				m_fw_total_input = new GTYPE[m_bw_buff_size];
#endif
	}

	void free_fw_input()
	{
#ifdef GPU
		if (m_fw_total_input)
		{
			HANDLE_ERROR(cudaFree(m_fw_total_input));
			m_fw_total_input = nullptr;
		}
#else
		if (m_fw_total_input)
		{
			delete[] m_fw_total_input;
			m_fw_total_input = nullptr;
		}
#endif
	}

	void create_fw_output()
	{
#ifdef GPU
			if (!m_fw_output)
				HANDLE_ERROR(cudaMalloc((void **)&m_fw_output, sizeof(GTYPE)*m_fw_buff_size));		
#else
			if (!m_fw_output)
				m_fw_output = new GTYPE[m_fw_buff_size];
#endif
	}

	void  create_bw_input()
	{
#ifdef GPU
			if (!m_bw_total_input)
				HANDLE_ERROR(cudaMalloc((void **)&m_bw_total_input, sizeof(GTYPE)*m_fw_buff_size));
#else
			if (!m_bw_total_input)
				m_bw_total_input = new GTYPE[m_fw_buff_size];
#endif
	}

	void free_bw_input()
	{
#ifdef GPU
		if (m_bw_total_input)
		{
			HANDLE_ERROR(cudaFree(m_bw_total_input));
			m_bw_total_input = nullptr;
		}
#else
		if (m_bw_total_input)
		{
			delete[] m_bw_total_input;
			m_bw_total_input = nullptr;
		}
#endif
	}

	void  create_bw_output()
	{
#ifdef GPU
			if (!m_bw_output)
				HANDLE_ERROR(cudaMalloc((void **)&m_bw_output, sizeof(GTYPE)*m_bw_buff_size));
		
#else
			if (!m_bw_output)
				m_bw_output = new GTYPE[m_bw_buff_size];
#endif
	}


	void free_prev_space()
	{
		for (int i = 0; i < m_prev_num; i++)
			m_prev_layer[i]->free_fw_output();
	}

	void free_next_space()
	{
		for (int i = 0; i < m_next_num; i++)
			m_next_layer[i]->free_bw_output();
	}

	//�õ�ǰ������
	bool get_fw_input()
	{
		if (m_prev_num < 1 ) return false;
		if (g_config.m_net_mode == RESNET) fw_input_resnet();
		else if (g_config.m_net_mode == DENSENET) fw_input_densenet();
#ifdef GPU
		/*//if (m_layer_num >= 78)
		{
			GTYPE  *str = new GTYPE[m_bw_buff_size];
			cudaMemcpy(str, m_fw_total_input, sizeof(GTYPE)*m_bw_buff_size, cudaMemcpyDeviceToHost);
			printf(" \n fw(%d)  %s\n", m_layer_num, m_layer_type);
			for (int i = 0; i < m_bw_buff_size; i++)
			{
				if (i % m_bw_node_size == 0) printf("\n");
				if (i % m_bw_col == 0) printf("\n");
				printf("%15f   ", str[i]);
			}
		}*/
#else

		/*{
			GTYPE  *str = new GTYPE[m_bw_buff_size];
			memcpy(str, m_fw_total_input, sizeof(GTYPE)*m_bw_buff_size);
			printf(" \n fw(%d)  %s\n", m_layer_num, m_layer_type);
			for (int i = 0; i < m_bw_buff_size; i++)
			{
				if (i % m_bw_node_size == 0) printf("\n");
				if (i % m_bw_col == 0) printf("\n");
				printf("%15f   ", str[i]);
			}
		}*/

#endif
		return true;
	}

	//�õ���������
	bool get_bw_input()
	{
		if ( m_next_num < 1) return false;		
		if (g_config.m_net_mode == RESNET) bw_input_resnet();
		else if (g_config.m_net_mode == DENSENET) bw_input_densenet();
#ifdef GPU
		/*//if (m_layer_num >= 78)
		{
		GTYPE  *str = new GTYPE[m_fw_buff_size];
		cudaMemcpy(str, m_bw_total_input, sizeof(GTYPE)*m_fw_buff_size, cudaMemcpyDeviceToHost);
		printf(" \n bw(%d)  %s\n", m_layer_num, m_layer_type);
		for (int i = 0; i < m_fw_batch_size; i++)
		{
		if (i % m_fw_node_size == 0) printf("\n");
		if (i % m_fw_col == 0) printf("\n");
		printf("%15f   ", str[i]);
		}
		}*/
#endif
		/*{
			GTYPE  *str = new GTYPE[m_fw_buff_size];
			memcpy(str, m_bw_total_input, sizeof(GTYPE)*m_fw_buff_size);
			printf(" \n bw(%d)  %s\n", m_layer_num, m_layer_type);
			for (int i = 0; i < m_fw_batch_size; i++)
			{
				if (i % m_fw_node_size == 0) printf("\n");
				if (i % m_fw_col == 0) printf("\n");
				printf("%15f   ", str[i]);
			}
		}*/
		return true;
	}

	GTYPE  regular_proc(GTYPE gd, GTYPE weight)
	{
		if ( g_config.m_regular_mode == L1_RULE)
		{
			//weight =0 ���ɵ�
			if (weight > 0)
				gd += g_config.m_rule_rate;
			else if (weight < 0)
				gd += g_config.m_rule_rate * -1.0;
		}
		else if (g_config.m_regular_mode == L2_RULE)
		{
			gd += g_config.m_rule_rate*weight;
		}
		return gd;
	}

	//�ݶ�У��ʱ��Ҫ�������
	GTYPE regular_check(GTYPE weight)
	{
		if (g_config.m_regular_mode == L1_RULE)
			return g_config.m_rule_rate*(abs(weight + INFINIT_NUM) - abs(weight - INFINIT_NUM));
		else if (g_config.m_regular_mode == L2_RULE)
			return 0.5*g_config.m_rule_rate*((weight + INFINIT_NUM)* (weight + INFINIT_NUM) - (weight - INFINIT_NUM)*(weight - INFINIT_NUM));
		return 0;		
	}

	void adam_study(GTYPE g, GTYPE* one, GTYPE* two, GTYPE *weight)
	{
		GTYPE gd = regular_proc(g, *weight);
		if (m_train_num == 1)
		{
			*one = *two = 0;
		}
		*one = MEAN_WEIGTH_RATE * *one + (1 - MEAN_WEIGTH_RATE)* gd;
		*two = VAR_WEIGTH_RATE * *two + (1 - VAR_WEIGTH_RATE)* gd * gd;
		GTYPE t1 = *one / (1 - pow(MEAN_WEIGTH_RATE, m_train_num));
		GTYPE t2 = *two / (1 - pow(VAR_WEIGTH_RATE, m_train_num));
		*weight = *weight - g_config.m_stu_rate *t1 / (sqrt(t2) + INFINIT_NUM);
		return;
	}

	void momentum_study(GTYPE g, GTYPE *weight, GTYPE *before)
	{
		GTYPE gd = regular_proc(g, *weight);
		if (m_train_num == 1) *before = 0;
		*before = *before * g_config.m_momentum_rate - gd * g_config.m_stu_rate;
		*weight = *weight + *before;
		return;
	}

	void static_study(GTYPE g, GTYPE *weight)
	{
		GTYPE gd = regular_proc(g, *weight);
		*weight = *weight - g_config.m_stu_rate*gd;
		return;
	}

	//�����麯��
	
	virtual void init_space() {}
	virtual void init(){}
	virtual void forward_proc(){}
	virtual void backword_proc(){}
	virtual void update_weight() {}
	virtual void create_test_weight(GTYPE(*f)(Clayer *start), Clayer *start){}
	virtual void check_weight(){}

	virtual void init_param(ifstream *f = nullptr){}
	virtual void save_param(ofstream *f) {}

	void clean_mem()
	{
		if (m_fw_total_input)
		{
#ifdef GPU
			HANDLE_ERROR(cudaFree(m_fw_total_input));
			m_fw_total_input = nullptr;
#else
			delete[] m_fw_total_input;
			m_fw_total_input = nullptr;
#endif
		}


		if (m_fw_output)
		{
#ifdef GPU
			HANDLE_ERROR(cudaFree(m_fw_output));
			m_fw_output = nullptr;
#else
			delete[] m_fw_output;
			m_fw_output = nullptr;
#endif
		}
	}


	//ѵ��һЩ��̬�仯�Ĳ���
	bool get_param()
	{
		return read_param();
	}
	void update_param(char *dir)
	{
		write_param(dir);
	}

	//�õ�ǰ������ߴ�
	void init_base_size()
	{
		if (m_prev_num > 0 )
		{
			if (g_config.m_net_mode == RESNET)
			{
				m_bw_node_num = m_prev_layer[0]->m_fw_node_num;
				m_bw_row = m_prev_layer[0]->m_fw_row;
				m_bw_col = m_prev_layer[0]->m_fw_col;
			}
			else if (g_config.m_net_mode == DENSENET)
			{
				m_bw_row = m_prev_layer[0]->m_fw_row;
				m_bw_col = m_prev_layer[0]->m_fw_col;
				m_bw_node_num = 0;
				for(int i = 0; i< m_prev_num; i++)
					m_bw_node_num += m_prev_layer[i]->m_fw_node_num;
			}
			if (m_fw_node_num == 0)
				m_fw_node_num = m_bw_node_num;
		}
	}

#ifdef GPU
	void set_const()
	{
		//g_cpu_layer[m_layer_num].batch_num = m_batch_num;		
		g_cpu_layer[m_layer_num].fw_row = m_fw_row;
		g_cpu_layer[m_layer_num].fw_col = m_fw_col;
		g_cpu_layer[m_layer_num].fw_real_row = m_fw_real_row;
		g_cpu_layer[m_layer_num].fw_real_col = m_fw_real_col;
		g_cpu_layer[m_layer_num].fw_node_num = m_fw_node_num;
		//g_cpu_layer[m_layer_num].fw_buff_size = m_fw_buff_size;
		g_cpu_layer[m_layer_num].fw_batch_size = m_fw_batch_size;
		g_cpu_layer[m_layer_num].fw_node_size = m_fw_node_size;
		g_cpu_layer[m_layer_num].bw_row = m_bw_row;
		g_cpu_layer[m_layer_num].bw_col = m_bw_col;
		//g_cpu_layer[m_layer_num].bw_buff_size = m_bw_buff_size;
		g_cpu_layer[m_layer_num].bw_batch_size = m_bw_batch_size;
		g_cpu_layer[m_layer_num].bw_node_size = m_bw_node_size;
		g_cpu_layer[m_layer_num].bw_node_num = m_bw_node_num;
	}
#endif

	void get_fw_output(GTYPE *out)
	{
		if (!m_fw_output) return;
#ifdef GPU
		HANDLE_ERROR(cudaMemcpy(out, m_fw_output, sizeof(GTYPE) *m_fw_buff_size, cudaMemcpyDeviceToHost));
#else
		memcpy(out, m_fw_output, sizeof(GTYPE) *m_fw_buff_size);
#endif
	}

	void free_fw_output()
	{
		m_fw_del_request++;
		if (m_fw_del_request < m_next_num) return;
		m_fw_del_request = 0;
#ifdef GPU
		if (m_fw_output)
		{
			HANDLE_ERROR(cudaFree(m_fw_output));
			m_fw_output = nullptr;
		}
#else
		if (m_fw_output)
		{
			delete[] m_fw_output;
			m_fw_output = nullptr;
		}
#endif
	}

	void free_bw_output()
	{
		m_bw_del_request++;
		if (m_bw_del_request < m_prev_num) return;
		m_bw_del_request = 0;

#ifdef GPU
		if (m_bw_output)
		{
			HANDLE_ERROR(cudaFree(m_bw_output));
			m_bw_output = nullptr;
		}
#else
		if (m_bw_output)
		{
			delete[] m_bw_output;
			m_bw_output = nullptr;
		}
#endif
	}

private:	

	bool write_param(char *dir)
	{
		ofstream f;
		char file[FILENAME_LEN];

		sprintf(file,"%s/%s", dir, GLOBAL_PARAM_FILE);
		if (!open_out_file(f, file)) return false;
		f << setprecision(PARAM_PRECISION) << endl;
		f << "mincost=" << " " << m_mincost << endl;
		f << "bestanswer=" << " " << m_best_answer << endl;
		f << "traincount=" << " " << m_total_train << endl;
		f.close();
		return true;
	}

	bool read_param()
	{
		ifstream f;
		char file[FILENAME_LEN];
		sprintf(file,"%s/%s", g_config.m_work_param_dir, GLOBAL_PARAM_FILE);

		if (!open_in_file(f, file)) return false;
		char tempstr[128];
		f >> tempstr >> m_mincost;
		f >> tempstr >> m_best_answer;
		f >> tempstr >> m_total_train;
		f.close();
		return true;
	}

	void fw_input_densenet()
	{
		//���뱣֤prev1 prev2 prev3��fw_row ��fw_colһ��
		int size;

#ifdef GPU
		struct gpu_t st;
		st.result = m_fw_total_input;
		st.result_layer = m_layer_num;

		unsigned long long in[MAX_CONN_NUM];
		memset(in, 0, sizeof(unsigned long long)*MAX_CONN_NUM);
		unsigned int in_layer[MAX_CONN_NUM];

		for (size = 0; size < m_prev_num; size++)
		{
			in[size] = (unsigned long long)m_prev_layer[size]->m_fw_output;
			in_layer[size] = m_prev_layer[size]->m_layer_num;
		}
		HANDLE_ERROR(cudaMemcpy(m_in_buf, in, sizeof(unsigned long long) *MAX_CONN_NUM, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(m_in_layer_buf, in_layer, sizeof(unsigned int) *MAX_CONN_NUM, cudaMemcpyHostToDevice));
		st.in = m_in_buf;
		st.in_layer = m_in_layer_buf;		
		st.start = 0;
		st.len = m_bw_buff_size;
		gpu_cal(st, FN_DENSENET_LAYER_FW_INPUT);
#else
		unsigned int len = 0;
		for (unsigned int i = 0; i < m_batch_num; i++)
		{
			for (size = 0; size < m_prev_num; size++)
			{
				memcpy(m_fw_total_input + len, m_prev_layer[size]->m_fw_output + i*m_prev_layer[size]->m_fw_batch_size, m_prev_layer[size]->m_fw_batch_size * sizeof(GTYPE));
				len += m_prev_layer[size]->m_fw_batch_size;					
			}
		}
#endif
	}

	void fw_input_resnet()
	{
		int size;

#ifdef GPU
		struct gpu_t st;
		st.result = m_fw_total_input;
		st.result_layer = m_prev_layer[0]->m_layer_num;

		unsigned long long in[MAX_CONN_NUM];
		memset(in, 0, sizeof(unsigned long long)*MAX_CONN_NUM);
		unsigned int in_layer[MAX_CONN_NUM];

		for (size = 0; size < m_prev_num; size++)
		{
			in[size] = (unsigned long long)m_prev_layer[size]->m_fw_output;
			in_layer[size] = m_prev_layer[size]->m_layer_num;
		}
		HANDLE_ERROR(cudaMemcpy(m_in_buf, in, sizeof(unsigned long long) *MAX_CONN_NUM, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(m_in_layer_buf, in_layer, sizeof(unsigned int) *MAX_CONN_NUM, cudaMemcpyHostToDevice));
		st.in = m_in_buf;
		st.in_layer = m_in_layer_buf;

		st.start = 0;
		st.len = m_prev_layer[0]->m_fw_buff_size;
		gpu_cal(st, FN_RESNET_LAYER_ADD);
#else
		unsigned int   i;

		for (size = 0; size < m_prev_num; size++)
		{
			if(size == 0)
				memcpy(m_fw_total_input, m_prev_layer[size]->m_fw_output, m_prev_layer[size]->m_fw_buff_size * sizeof(GTYPE));
			else
			{
				if( m_prev_layer[size]->m_fw_buff_size == m_prev_layer[0]->m_fw_buff_size)
				{
					for (i = 0; i < m_prev_layer[0]->m_fw_buff_size; i++)
						m_fw_total_input[i] += m_prev_layer[size]->m_fw_output[i];
				}
				else if (m_prev_layer[size]->m_fw_buff_size > m_prev_layer[0]->m_fw_buff_size)
				{
					//�ߴ粻һ����ͨ�����ǲв���·��ǰ��ĵĻ�������֧����С�ġ��Ǵ�ߴ�һ�롣���պ����������ģ���ȡ��2���
					for (i = 0; i < m_prev_layer[0]->m_fw_buff_size; i++)
					{
						int batch_x = i / m_prev_layer[0]->m_fw_batch_size;
						int batch_y = i % m_prev_layer[0]->m_fw_batch_size;
						int node_x = batch_y / m_prev_layer[0]->m_fw_node_size;
						int node_y = batch_y % m_prev_layer[0]->m_fw_node_size;

						int x = node_y / m_prev_layer[0]->m_fw_col;
						int y = node_y % m_prev_layer[0]->m_fw_col;
						if (2 * x < m_prev_layer[size]->m_fw_row && 2 * y < m_prev_layer[size]->m_fw_col)
							m_fw_total_input[i] +=
							m_prev_layer[size]->m_fw_output[batch_x*m_prev_layer[size]->m_fw_batch_size + node_x*m_prev_layer[size]->m_fw_node_size + (2 * x)*m_prev_layer[size]->m_fw_col + 2 * y];
					}
				}
				else /*if (m_prev_layer2->m_buff_size < m_prev_layer1->m_buff_size)*/
				{
					//��δ���û���ԣ�ʵ���м���������֡�ͨ��������֧1����С
					for (i = 0; i < m_prev_layer[size]->m_fw_buff_size; i++)
					{
						int batch_x = i / m_prev_layer[size]->m_fw_batch_size;
						int batch_y = i % m_prev_layer[size]->m_fw_batch_size;
						int node_x = batch_y / m_prev_layer[size]->m_fw_node_size;
						int node_y = batch_y % m_prev_layer[size]->m_fw_node_size;

						int x = node_y / m_prev_layer[size]->m_fw_col;
						int y = node_y % m_prev_layer[size]->m_fw_col;
						if (2 * x < m_prev_layer[0]->m_fw_row && 2 * y < m_prev_layer[0]->m_fw_col)
							m_fw_total_input[batch_x*m_prev_layer[0]->m_fw_batch_size + node_x*m_prev_layer[0]->m_fw_node_size + (2 * x)*m_prev_layer[0]->m_fw_col + 2 * y] +=
							m_prev_layer[size]->m_fw_output[i];
					}
				}
			}
		}
#endif
	}

	void bw_input_densenet()
	{
		unsigned int size;

#ifdef GPU
		struct gpu_t st;
		st.result = m_bw_total_input;	
		st.result_layer = m_layer_num;

		unsigned long long in[MAX_CONN_NUM];
		memset(in, 0, sizeof(unsigned long long)*MAX_CONN_NUM);
		unsigned int in_layer[MAX_CONN_NUM];
		unsigned int in_pos[MAX_CONN_NUM];

		for (size = 0; size < m_next_num; size++)
		{
			in[size] = (unsigned long long)m_next_layer[size]->m_bw_output;
			in_layer[size] = m_next_layer[size]->m_layer_num;
			in_pos[size] = 0;
			for (int n = 0; n < m_next_pos[size]; n++)
				in_pos[size] += m_next_layer[size]->m_prev_layer[n]->m_fw_batch_size;
		}
		HANDLE_ERROR(cudaMemcpy(m_in_buf, in, sizeof(unsigned long long) *MAX_CONN_NUM, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(m_in_layer_buf, in_layer, sizeof(unsigned int) *MAX_CONN_NUM, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(m_in_pos_buf, in_pos, sizeof(unsigned int) *MAX_CONN_NUM, cudaMemcpyHostToDevice));
		st.in = m_in_buf;
		st.in_layer = m_in_layer_buf;
		st.in_pos = m_in_pos_buf;
		st.start = 0;
		st.len = m_fw_buff_size;
		gpu_cal(st, FN_DENSENET_LAYER_BW_INPUT);
#else
		unsigned int i, len;
		for (size = 0; size < m_batch_num; size++)
		{
			for (int m = 0; m < m_next_num; m++)
			{
				len = 0;
				for (int n = 0; n < m_next_pos[m]; n++)
					len += m_next_layer[m]->m_prev_layer[n]->m_fw_batch_size;
				if (m == 0)
					memcpy(m_bw_total_input + size*m_fw_batch_size, m_next_layer[m]->m_bw_output + size*m_next_layer[m]->m_bw_batch_size + len,
						m_fw_batch_size * sizeof(GTYPE));
				else
				{
					for (i = 0; i < m_fw_batch_size; i++)
						m_bw_total_input[size*m_fw_batch_size + i] += m_next_layer[m]->m_bw_output[size*m_next_layer[m]->m_bw_batch_size + len + i];
				}
			}
		}
#endif
	}


	void bw_input_resnet()
	{
		unsigned int size;

#ifdef GPU
		struct gpu_t st;
		st.result = m_bw_total_input;
		st.result_layer = m_layer_num;

		unsigned long long in[MAX_CONN_NUM];
		memset(in, 0, sizeof(unsigned long long)*MAX_CONN_NUM);
		unsigned int in_layer[MAX_CONN_NUM];

		for (size = 0; size < m_next_num; size++)
		{
			in[size] = (unsigned long long)m_next_layer[size]->m_bw_output;
			in_layer[size] = m_next_layer[size]->m_prev_layer[0]->m_layer_num;
		}
		HANDLE_ERROR(cudaMemcpy(m_in_buf, in, sizeof(unsigned long long) *MAX_CONN_NUM, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(m_in_layer_buf, in_layer, sizeof(unsigned int) *MAX_CONN_NUM, cudaMemcpyHostToDevice));
		st.in = m_in_buf;
		st.in_layer = m_in_layer_buf;	
		st.start = 0;
		st.len = m_fw_buff_size;
		gpu_cal(st, FN_RESNET_LAYER_ADD);
#else
		unsigned int   i;
		for (size = 0; size < m_next_num; size++)
		{
			if (m_next_layer[size]->m_prev_layer[0]->m_fw_buff_size == m_fw_buff_size)
			{
				if(size == 0)
					memcpy(m_bw_total_input, m_next_layer[size]->m_bw_output, m_fw_buff_size * sizeof(GTYPE));
				else
				{
					for (i = 0; i < m_fw_buff_size; i++)
						m_bw_total_input[i] += m_next_layer[size]->m_bw_output[i];
				}
			}
			else if (m_next_layer[size]->m_prev_layer[0]->m_fw_buff_size < m_fw_buff_size)
			{
				if(size == 0)
					memset(m_bw_total_input, 0, m_fw_buff_size * sizeof(GTYPE));
				//�ߴ粻һ����ͨ�����ǲв���·���Ǵ�ߴ�һ�롣���պ����������ģ���ȡ��2���
				for (i = 0; i < m_next_layer[size]->m_prev_layer[0]->m_fw_buff_size; i++)
				{
					int batch_x = i / m_next_layer[size]->m_prev_layer[0]->m_fw_batch_size;
					int batch_y = i % m_next_layer[size]->m_prev_layer[0]->m_fw_batch_size;
					int node_x = batch_y / m_next_layer[size]->m_prev_layer[0]->m_fw_node_size;
					int node_y = batch_y % m_next_layer[size]->m_prev_layer[0]->m_fw_node_size;

					int x = node_y / m_next_layer[size]->m_prev_layer[0]->m_fw_col;
					int y = node_y % m_next_layer[size]->m_prev_layer[0]->m_fw_col;
					if (2 * x < m_fw_row && 2 * y < m_fw_col)
					{
						if(size == 0)
							m_bw_total_input[batch_x*m_fw_batch_size + node_x*m_fw_node_size + (2 * x)*m_fw_col + 2 * y] =	m_next_layer[size]->m_bw_output[i];
						else
							m_bw_total_input[batch_x*m_fw_batch_size + node_x*m_fw_node_size + (2 * x)*m_fw_col + 2 * y] += m_next_layer[size]->m_bw_output[i];
					}
				}
			}
			else /*if (m_next_layer1->m_prev_layer1->m_buff_size > m_buff_size)*/
			{
				//û�в�����δ��룬����ͨ���ǲ��������һ���
				for (i = 0; i < m_fw_buff_size; i++)
				{
					int batch_x = i / m_fw_batch_size;
					int batch_y = i % m_fw_batch_size;
					int node_x = batch_y / m_fw_node_size;
					int node_y = batch_y % m_fw_node_size;

					int x = node_y / m_fw_col;
					int y = node_y % m_fw_col;

					if (2 * x < m_next_layer[size]->m_prev_layer[0]->m_fw_row && 2 * y < m_next_layer[size]->m_prev_layer[0]->m_fw_col)
					{
						if(size == 0)
							m_bw_total_input[i] = m_next_layer[size]->m_bw_output[batch_x*m_next_layer[size]->m_prev_layer[0]->m_fw_batch_size +
							node_x*m_next_layer[size]->m_prev_layer[0]->m_fw_node_size + (2 * x)*m_next_layer[size]->m_prev_layer[0]->m_fw_col + 2 * y];
						else
							m_bw_total_input[i] += m_next_layer[size]->m_bw_output[batch_x*m_next_layer[size]->m_prev_layer[0]->m_fw_batch_size +
							node_x*m_next_layer[size]->m_prev_layer[0]->m_fw_node_size + (2 * x)*m_next_layer[size]->m_prev_layer[0]->m_fw_col + 2 * y];
					}
				}
			}
		}
#endif
	}
};
unsigned int  Clayer::m_train_num = 0;  //ѵ���˵Ĵ���
unsigned int  Clayer::m_batch_num = 1; //����������Ŀ
int Clayer::m_net_status = CONT_TRAINING; //������ǰ��ѵ��
GTYPE  Clayer::m_mincost = 65535.643289642394629;
unsigned int  Clayer::m_total_train = 0;  //ѵ���˵Ĵ���
GTYPE  Clayer::m_best_answer = 65535.643289642394629;
#endif