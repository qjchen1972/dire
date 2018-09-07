/*
* Copyright (c) 2017
* All rights reserved.
*
* 文件名称: active.h
* 文件标识:
* 摘    要:激励函数，用于线性和非线性函数之间的转换
*
*
* 当前版本: 1.0
* 作    者: chen.qian.jiang
* 开始日期: 2017-06-15
* 完成日期:
* 其它说明：
* 修改日期        版本号     修改人          修改内容
* -----------------------------------------------
* 2017/06/22      V1.0       陈黔江          创建
* 2017/07/15      V1.0       陈黔江          增加gpu
*/
#pragma once
#ifndef _ACTIVE__H
#define _ACTIVE__H


#include "layer.h"

class Active : public Clayer
{
public:
	Active() 
	{		
		m_layer_type = ACIVE_LAYER;	
	}
	~Active() {}

	
	void init_space()
	{
#ifdef GPU
		set_const();
		if (m_active_type == LEAKY_RELU)
		{
			HANDLE_ERROR(cudaMalloc((void **)&m_leakey_relu_gpu, sizeof(GTYPE)));
			HANDLE_ERROR(cudaMalloc((void **)&m_leakey_relu_diff_gpu, sizeof(GTYPE)));
		}
		else if (m_active_type == SOFTMAX)
		{
			HANDLE_ERROR(cudaMalloc((void **)&m_softmax_total, sizeof(GTYPE)*m_batch_num));
		}
#else
		if (m_active_type == SOFTMAX)
		{
			m_softmax_total = new GTYPE[m_batch_num];
		}
#endif
	}	
	
	void init()
	{
		init_size();
	}

	void init_param(ifstream *f = nullptr)
	{
		if (m_active_type != LEAKY_RELU) return;		

		if (m_net_status == PARAM_CHECK)
			sprintf(m_check_file, "%s%d.txt", ACTIVE_CHECK_LOG, m_layer_num);
	
		if (m_net_status == CONT_TRAINING || m_net_status == TESTING)
		{
			read_param(f);
			return;
		}

		//初始化weight
		//y = max(0,x) + a * min(0,x)
		//因为:
		//var(yk）= 1/2 * var(yk-1)   x>= 0
		// var(yk) = 1/2 * var(a) *var(yk-1)  x<0
		//所以:
		//var(yk) = (1/2 + 1/2* var(a)) *var(yk-1)
		//var(a) =1 就可以了
		if (g_config.m_initweight_mode == XAVIER)
		{
			m_leakey_relu = uniform_rand(-sqrt(3), sqrt(3));
		}
		else if (g_config.m_initweight_mode == MARS)
		{
			m_leakey_relu = gauss_rand(0, 1);
		}
		else
		{
			m_leakey_relu = gauss_rand(0, 1);
		}
	}

	void save_param(ofstream *f) 
	{
		if (m_active_type != LEAKY_RELU) return;
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
		if (m_active_type == SOFTMAX)
		{
			struct gpu_t st;
			st.in1 = m_fw_total_input;
			st.start = 0;
			st.len = m_bw_batch_size;
			st.sum_result = m_softmax_total;
			st.sum_result_count = m_batch_num;
			gpu_sum(st, FN_ACTIVE_FW_SUM);
		}
		struct gpu_t st;
		st.result = m_fw_output;
		st.result_layer = m_layer_num;
		st.in1 = m_fw_total_input;
		if (m_active_type == SOFTMAX)
			st.in2 = m_softmax_total;
		else if (m_active_type == LEAKY_RELU)
			st.in2 = m_leakey_relu_gpu;		
		st.start = 0;
		st.len = m_fw_buff_size;
		gpu_cal(st, FN_ACTIVE_FW);

#else
		unsigned int   i,size;

		if (m_active_type == SOFTMAX)
		{
			for (size = 0; size < m_batch_num; size++)
			{
				m_softmax_total[size] = 0.0;
				unsigned int   batch_pos = size * m_bw_batch_size;
				//先求出sigma					
				for (i = 0; i < m_bw_batch_size; i++)
					m_softmax_total[size] += exp(m_fw_total_input[batch_pos + i]);
			}			
		}
		for (i = 0; i < m_fw_buff_size; i++)
		{
			if (m_active_type == SOFTMAX)
				m_fw_output[i] = active(m_active_type, m_fw_total_input[i], m_softmax_total[i / m_fw_batch_size]);
			else
				m_fw_output[i] = active(m_active_type, m_fw_total_input[i], m_leakey_relu);
		}		
#endif

		if (g_config.m_free_type != NERVER_FREE)
		{
			//让backward来删除
			if (m_active_type != RELU && m_active_type != LEAKY_RELU) m_fw_del_request--;
			//用完立即删
			if (g_config.m_free_type == FUNC_FREE)		free_fw_input();
		}
	}				

	//softmax可以推导为以下公式 
	//df/dxi = yi * (sigma( dE/dyk *( -yk))  + dE / dyi  * yi
	void backword_proc()
	{	
		if (m_next_num <= 0 || m_prev_num <= 0) return;
		create_bw_input();
		get_bw_input();		
		create_bw_output();
		if (!m_fw_total_input)
		{
			create_fw_input();
			get_fw_input();
		}

#ifdef GPU
		if (m_active_type == SOFTMAX)
		{			
			struct gpu_t st;
			st.in1 = m_bw_total_input;
			st.in2 = m_fw_output;
			st.start = 0;
			st.len = m_fw_batch_size;
			st.sum_result = m_softmax_total;
			st.sum_result_count = m_batch_num;
			gpu_sum(st, FN_ACTIVE_BW_SUM);
		}

		struct gpu_t st;
		st.result = m_bw_output;
		st.in1 = m_fw_total_input;
		st.in2 = m_fw_output;
		st.in2_layer = m_layer_num;
		st.in3 = m_bw_total_input;
		if (m_active_type == SOFTMAX)
			st.in4 = m_softmax_total;
		else if (m_active_type == LEAKY_RELU)
			st.in4 = m_leakey_relu_gpu;		
		st.start = 0;
		st.len = m_bw_buff_size;			
		gpu_cal(st,FN_ACTIVE_BW);		
#else	
		unsigned int   i;

		if (m_active_type == SOFTMAX)
		{
			int size;
			for (size = 0; size < m_batch_num; size++)
			{
				m_softmax_total[size] = 0.0;
				int batch_pos = size * m_fw_batch_size;
				for (i = 0; i < m_fw_batch_size; i++)
					m_softmax_total[size] += m_bw_total_input[batch_pos + i] * m_fw_output[batch_pos + i];				
				m_softmax_total[size] = -1.0 * m_softmax_total[size];
			}
		}

		for (i = 0; i < m_bw_buff_size; i++)
		{		
			if (m_active_type == SOFTMAX)
				m_bw_output[i] = active_derivation(m_active_type, m_fw_total_input[i], m_fw_output[i], m_bw_total_input[i],m_softmax_total[i/ m_bw_batch_size]);
			else
			{
				if(m_fw_output)
					m_bw_output[i] = active_derivation(m_active_type, m_fw_total_input[i], m_fw_output[i], m_bw_total_input[i], m_leakey_relu);
				else
					m_bw_output[i] = active_derivation(m_active_type, m_fw_total_input[i], 0, m_bw_total_input[i], m_leakey_relu);
			}
		}
#endif
		if (g_config.m_free_type != NERVER_FREE)
		{
			//我来删除fw_output
			if ( m_active_type != RELU && m_active_type != LEAKY_RELU ) free_fw_output();
			if ( m_active_type != LEAKY_RELU )
			{
				free_fw_input();
				free_bw_input();
				free_prev_space();
				free_next_space();
			}			
			else
			{
				if (g_config.m_free_type == FUNC_FREE)
				{
					free_fw_input();
					free_bw_input();
				}
			}
		}
	}

	//只有采用了LEAKY_RELU才会有权值更新
	// 若是m_leaky_relu全局都采用同一个参数：
	//m_leaky_relu的权值更新需要所有使用了LEAKY_RELU的layer的残差
	//若是采用动量的学习率，需要全局进行权值更新计算。
	//本API库对这种牵涉全局的残差的权值更新，都采用非动量的学习率，用来保证可以分层独立的更新
	//如 第一层有 df/dr  = df/dr - temp0 第2层  也是 df/dr = df/dr - temp1,那么就有 df/dr = df/dr -(temp0+temp1) 
	//-------------------------------
	//但是，思来想去，决定这儿采用每一层使用不同的m_leaky_relu
	void update_weight()
	{
		if (m_active_type != LEAKY_RELU) return;
		get_diff();
		if (g_config.m_study_mode == ADAM_STUDY)
			adam_study(m_leakey_relu_diff, &m_leakey_relu_one, &m_leakey_relu_two, &m_leakey_relu);
		else if (g_config.m_study_mode == MOMENTUM_STUDY)
			momentum_study(m_leakey_relu_diff, &m_leakey_relu, &m_leakey_relu_diff_before);
		else
			static_study(m_leakey_relu_diff, &m_leakey_relu);
	}	

	//生成梯度校验初值
	void create_test_weight(GTYPE(*f)(Clayer *start), Clayer *start)
	{
		if (m_active_type != LEAKY_RELU) return;
		GTYPE temp;
		GTYPE err1, err2;
	
		temp = m_leakey_relu;
		m_leakey_relu = temp - INFINIT_NUM;
		err1 = f(start);

		m_leakey_relu = temp + INFINIT_NUM;
		err2 = f(start);

		m_leakey_relu_check = (err2 - err1 + regular_check(temp)) / (2.0 * INFINIT_NUM);
		m_leakey_relu = temp;
	}

	//校验梯度，任何新算法必须做这校验
	void check_weight()
	{
		if (m_active_type != LEAKY_RELU) return;				
	
		get_diff();
		ofstream f;
		if (open_out_file(f, m_check_file))
		{
			f << setprecision(PARAM_PRECISION) << m_leakey_relu_check << " " << regular_proc(m_leakey_relu_diff, m_leakey_relu) << endl;
			f.close();
		}
	}

	

	void set_atcive_type(int type){	m_active_type = type;}
	int  get_atcive_type() { return m_active_type;}
	GTYPE get_leakey_relu(){ return m_leakey_relu; }	

private:	
	
	//激励函数
	GTYPE active(int type, GTYPE src, GTYPE param = 0)
	{
		GTYPE answer = 0.0;
		if (type == RELU)
		{
			answer = src >= 0 ? src : 0;
		}
		else if (type == SIGMOID)
		{
			answer = 1.0 / (1.0 + exp(-1.0 *  src));
		}
		else if (type == TANH)
		{
			answer = (exp(src) - exp(-1.0*src)) / (exp(src) + exp(-1.0*src));
		}
		else if (type == SOFTMAX)
		{
			answer = exp(src) / param;
		}
		else if (type == LEAKY_RELU)
		{
			answer = src >= 0 ? src : param*src;
		}
		return answer;
	}


	//利用激励函数的导数计算,从而得到了后向输出。不是激励函数的导数值，是后向输出！！！
	GTYPE active_derivation(int type, GTYPE in, GTYPE out, GTYPE loss_in, GTYPE param = 0)
	{
		GTYPE answer = 0.0;

		if (type == RELU)
		{
			answer = in >= 0 ? loss_in : 0;
		}
		else if (type == SIGMOID)
		{
			answer = out * (1 - out) * loss_in;
		}
		else if (type == TANH)
		{
			answer = (1.0 - out * out) * loss_in;
		}
		else if (type == SOFTMAX)
		{
			answer = out*param + loss_in * out;
		}
		else if (type == LEAKY_RELU)
		{	
			//in =0 时？ 应该是不可导
			answer = in >= 0 ? loss_in : param * loss_in;
		}
		return answer;
	}

	bool write_param(ofstream *f)
	{		
		(*f).write((char*)(&m_leakey_relu), sizeof(GTYPE));
		return true;
	}

	bool read_param(ifstream *f)
	{
		(*f).read((char*)(&m_leakey_relu), sizeof(GTYPE));
		if ((*f).eof()) return false;
		return true;
	}

	void  set_param()
	{
		if (m_active_type != LEAKY_RELU) return;
#ifdef GPU
		HANDLE_ERROR(cudaMemcpy(m_leakey_relu_gpu, &m_leakey_relu, sizeof(GTYPE), cudaMemcpyHostToDevice));
#endif
	}

#ifdef GPU
	void set_const()
	{
		g_cpu_layer[m_layer_num].type.active.active_type = m_active_type;
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

	void get_diff()
	{
		if (m_next_num <= 0 || m_prev_num <= 0) return;
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
		st.in1 = m_fw_total_input;
		st.start = 0;
		st.len = m_fw_buff_size;
		st.sum_result = m_leakey_relu_diff_gpu;
		st.sum_result_count = 1;
		st.batchsize = m_batch_num;
		gpu_sum(st, FN_ACTIVE_UPDATE_SUM);
		HANDLE_ERROR(cudaMemcpy(&m_leakey_relu_diff, m_leakey_relu_diff_gpu, sizeof(GTYPE), cudaMemcpyDeviceToHost));
#else
		unsigned int   i;
		m_leakey_relu_diff = 0;

		for (i = 0; i < m_fw_buff_size; i++)
		{
			if (m_fw_total_input[i] < 0)
				m_leakey_relu_diff += m_bw_total_input[i] * m_fw_total_input[i];
		}
		m_leakey_relu_diff = m_leakey_relu_diff / m_batch_num;
#endif
		if (g_config.m_free_type != NERVER_FREE)
		{
			free_fw_input();
			free_bw_input();
			free_prev_space();
			free_next_space();
		}
	}

	//leaky relu系数，可以学习
	GTYPE m_leakey_relu = 0;
	//用于梯度校验
	GTYPE m_leakey_relu_check = 0;
	
	//导数
	GTYPE m_leakey_relu_diff = 0;

	//用于ADAM学习
	GTYPE m_leakey_relu_one = 0; //一阶矩
	GTYPE m_leakey_relu_two = 0; //二阶矩

	//用于冲量学习
	GTYPE m_leakey_relu_diff_before = 0; //前一导数
	
	//激励类型
	int  m_active_type = RELU;
	//用于计算的softmax，每一个batch需要计算其exp的累计和
	GTYPE *m_softmax_total = nullptr;

#ifdef GPU
	GTYPE *m_leakey_relu_gpu = nullptr;
	GTYPE *m_leakey_relu_diff_gpu = nullptr;
#endif	
};
#endif