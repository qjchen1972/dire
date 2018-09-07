/*
* Copyright (c) 2017
* All rights reserved.
*
* 文件名称: end.h
* 文件标识:
* 摘    要: 结束层
*
*
* 当前版本: 1.0
* 作    者: chen.qian.jiang
* 开始日期: 2017-06-14
* 完成日期:
* 其它说明：
* 修改日期        版本号     修改人          修改内容
* -----------------------------------------------
* 2017/06/19      V1.0       陈黔江          创建
* 2017/07/17      V1.0       陈黔江          增加gpu
*/
#pragma once
#ifndef _END__H
#define _END__H

#include "layer.h"


class End : public Clayer
{
public:
	End() 
	{ 
		m_layer_type = END_LAYER; 
	}
	~End() {}

	void init_space()
	{
#ifdef GPU
		set_const();
		HANDLE_ERROR(cudaMalloc((void **)&m_label, sizeof(GTYPE) *m_fw_buff_size));		
		HANDLE_ERROR(cudaMalloc((void **)&m_cost, sizeof(GTYPE)));
		if (m_cross_weight_first)
		{
			HANDLE_ERROR(cudaMalloc((void **)&m_label_weight_first, sizeof(GTYPE) *m_fw_batch_size));
			HANDLE_ERROR(cudaMemcpy(m_label_weight_first, m_cross_weight_first, sizeof(GTYPE) *m_fw_batch_size, cudaMemcpyHostToDevice));
    	}
		if (m_cross_weight_second)
		{
			HANDLE_ERROR(cudaMalloc((void **)&m_label_weight_second, sizeof(GTYPE) *m_fw_batch_size));
			HANDLE_ERROR(cudaMemcpy(m_label_weight_second, m_cross_weight_second, sizeof(GTYPE) *m_fw_batch_size, cudaMemcpyHostToDevice));
		}
#else
		m_label = new GTYPE[m_fw_buff_size];
		if (m_cross_weight_first)
		{
			m_label_weight_first = new GTYPE[m_fw_batch_size];
			memcpy(m_label_weight_first, m_cross_weight_first, sizeof(GTYPE) *m_fw_batch_size);
		}
		if (m_cross_weight_second)
		{
			m_label_weight_second = new GTYPE[m_fw_batch_size];
			memcpy(m_label_weight_second, m_cross_weight_second, sizeof(GTYPE) *m_fw_batch_size);
		}
#endif
		m_in = new GTYPE[m_fw_buff_size];
		if (m_cross_weight_first)
		{
			delete[] m_cross_weight_first;
			m_cross_weight_first = nullptr;
		}
		if (m_cross_weight_second)
		{
			delete[] m_cross_weight_second;
			m_cross_weight_second = nullptr;
		}
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
	}

	void backword_proc()
	{
		m_curcost = get_cost();
		if (fpclassify(m_curcost) == FP_NAN || fpclassify(m_curcost) == FP_INFINITE || fpclassify(m_curcost) == FP_SUBNORMAL)
		{
			printf("cost is a NAN\n");
			exit(EXIT_FAILURE);
		}
		if (m_train_num == 1) m_wholecost = 0;
		m_wholecost = (m_wholecost*(m_train_num - 1) + m_curcost)/m_train_num;
		printf("it is %5f   %d.  whole cost is  %5f\n", m_curcost, m_train_num, m_wholecost);

		create_bw_output();

#ifdef GPU
		struct gpu_t st;
		st.result = m_bw_output;
		st.in1 = m_fw_total_input;
		st.in2 = m_label;
		st.in2_layer = m_layer_num;
		if (m_cost_type == WEIGHT_CROSS_ENTROPY)
		{
			st.in3 = m_label_weight_first;
			st.in4 = m_label_weight_second;
		}
		st.start = 0;
		st.len = m_bw_buff_size;
		gpu_cal(st, FN_END_BW);
#else		
		unsigned int   i;
		for (i = 0; i < m_bw_buff_size; i++)
		{
			if (m_cost_type == LSE)
			{
				m_bw_output[i] = m_fw_total_input[i] - m_label[i];
			}
			else if (m_cost_type == CROSS_ENTROPY)
			{
				m_bw_output[i] = (1 - m_label[i]) / (1 - m_fw_total_input[i]) - m_label[i] / m_fw_total_input[i]; 
				//(m_fw_total_input[i] - m_label[i]) / (m_fw_total_input[i] * (1 - m_fw_total_input[i]));
			}
			else if (m_cost_type == MSE)
			{
				m_bw_output[i] = (m_fw_total_input[i] - m_label[i]) / m_fw_node_size;
			}
			else if (m_cost_type == WEIGHT_CROSS_ENTROPY)
			{
				unsigned int batch_y = i % m_fw_batch_size;
				m_bw_output[i] = m_label_weight_second[batch_y]*(1 - m_label[i]) / (1 - m_fw_total_input[i]) - m_label_weight_first[batch_y] * m_label[i] / m_fw_total_input[i];
			}			
			else
				m_bw_output[i] = 0;
		}
#endif

		if (g_config.m_free_type != NERVER_FREE)
		{
			free_fw_input();
			free_prev_space();
		}
	}

	void set_cost_type(int type)
	{
		m_cost_type = type; 
	}

	//得到cost值,需要前向执行完，才能执行这函数
	GTYPE get_cost()
	{
		GTYPE answer = 0;

#ifdef GPU
		/*
		GTYPE *input = new GTYPE[m_bw_buff_size];
		cudaMemcpy(input, m_fw_total_input, sizeof(GTYPE)*m_bw_buff_size, cudaMemcpyDeviceToHost);
		for(int i =0; i< m_bw_buff_size;i++)
		{
		if(i % m_bw_batch_size == 0 ) printf("\n");
		if(i %10 == 0 ) printf("\n");
		printf("(%f, %f),",m_in[i],input[i]);
		}
		printf("\n");
		*/
		struct gpu_t st;	
		st.result = m_label;
		st.result_layer = m_layer_num;
		st.in1 = m_fw_total_input;
		if (m_cost_type == WEIGHT_CROSS_ENTROPY)
		{
			st.in2 = m_label_weight_first;
			st.in3 = m_label_weight_second;
		}
		st.start = 0;
		st.len = m_bw_buff_size;
		st.sum_result = m_cost;
		st.sum_result_count = 1;
		st.batchsize = m_batch_num;
		gpu_sum(st, FN_END_COST_SUM);		
		HANDLE_ERROR(cudaMemcpy(&answer, m_cost,  sizeof(GTYPE) , cudaMemcpyDeviceToHost));
#else
		unsigned int    i;
		for (i = 0; i < m_bw_buff_size; i++)
		{
			if (m_cost_type == LSE)
			{
				answer += (m_label[i] - m_fw_total_input[i])*(m_label[i] - m_fw_total_input[i]) / 2.0;
			}
			else if (m_cost_type == CROSS_ENTROPY)
			{
				answer +=  -1.0 * (m_label[i] * log(m_fw_total_input[i]) + (1 - m_label[i])*log(1 - m_fw_total_input[i]));
			}
			else if(m_cost_type == MSE)
			{
				answer += (m_label[i] - m_fw_total_input[i])*(m_label[i] - m_fw_total_input[i]) /(2.0*m_fw_node_size);
			}
			else if(m_cost_type == WEIGHT_CROSS_ENTROPY)
			{
				unsigned int batch_y = i % m_fw_batch_size;
				answer += -1.0 * (m_label_weight_first[batch_y] * m_label[i] * log(m_fw_total_input[i]) +
					m_label_weight_second[batch_y] *(1 - m_label[i])*log(1 - m_fw_total_input[i]));
			}
		}
		answer = answer / m_batch_num;
#endif
		return  answer;
	}

	//set label
	void set_label(GTYPE *in)
	{
#ifdef GPU
		HANDLE_ERROR(cudaMemcpy(m_label, in,  sizeof(GTYPE) *m_fw_buff_size, cudaMemcpyHostToDevice));
#else
		memcpy(m_label, in,  sizeof(GTYPE) *m_fw_buff_size);
#endif
	}

	void set_label(char *buf, int type)
	{
		if (type == sizeof(GTYPE))
		{
			memcpy(m_in, buf, m_fw_buff_size * sizeof(GTYPE));
			set_label(m_in);
			return;
		}

		switch (type)
		{
		case 1:
		{
			unsigned char value;
			for (int i = 0; i < m_fw_buff_size; i++)
			{
				memcpy(&value, buf + i*sizeof(char), sizeof(char));
				m_in[i] = value;
			}
		}
		break;
		case 2:
		{
			short value;
			for (int i = 0; i < m_fw_buff_size; i++)
			{
				memcpy(&value, buf + i * sizeof(short), sizeof(short));
				m_in[i] = value;
			}
		}
		break;
		case 4:
		{
			float value;
			for (int i = 0; i < m_fw_buff_size; i++)
			{
				memcpy(&value, buf + i * sizeof(float), sizeof(float));
				m_in[i] = value;
			}
		}
		break;
		case 8:
		{
			float value;
			for (int i = 0; i < m_fw_buff_size; i++)
			{
				memcpy(&value, buf + i * sizeof(double), sizeof(double));
				m_in[i] = value;
			}
		}
		break;
		default:
			memcpy(m_in, buf, m_fw_buff_size * sizeof(GTYPE));
		}
		set_label(m_in);
	}


	void set_label_weight(GTYPE *in, int len )
	{
		m_cross_weight_first = new GTYPE[len];
		m_cross_weight_second = new GTYPE[len];

		for (int m = 0; m < len; m++)
		{
			m_cross_weight_first[m] = 1 - in[m];
			m_cross_weight_second[m] = in[m];
		}
	}

	void set_input_type(int type) { m_input_type = type; }

	GTYPE m_wholecost = 0;
	GTYPE m_curcost = 0;
	int  m_input_type = sizeof(GTYPE);
	GTYPE *m_in = nullptr;

private:
	

#ifdef GPU
	void set_const()
	{
		g_cpu_layer[m_layer_num].type.end.cost_type = m_cost_type;
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


	int m_cost_type = LSE;
	GTYPE *m_label = nullptr;  //样本
	GTYPE *m_label_weight_first = nullptr;//标签的权值
	GTYPE *m_label_weight_second = nullptr;//标签的权值

	GTYPE *m_cross_weight_first = nullptr;
	GTYPE *m_cross_weight_second = nullptr;	

#ifdef GPU
	GTYPE *m_cost = nullptr;	
#endif

};

#endif