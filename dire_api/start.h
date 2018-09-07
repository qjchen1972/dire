/*
* Copyright (c) 2017
* All rights reserved.
*
* �ļ�����: start.h
* �ļ���ʶ:
* ժ    Ҫ: ��ʼ��
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
* 2017/07/13      V1.0       ��ǭ��          ����gpu
*/
#pragma once
#ifndef _START__H
#define _START__H

#include "layer.h"

class Start : public Clayer
{
public:
	Start() 
	{
		m_layer_type = START_LAYER;
	}

	~Start() {}	

	void init_space()
	{

#ifdef GPU
		set_const();	
#endif
		m_in = new GTYPE[m_fw_buff_size];
	}

	void init()
	{
		init_size();
	}		

	void set_input(GTYPE *in)
	{
		if (m_net_status != TESTING && m_data_augm)
		{
			data_augm(in);
		}
		create_fw_output();

#ifdef GPU
		HANDLE_ERROR(cudaMemcpy(m_fw_output, in, m_fw_buff_size *  sizeof(GTYPE) , cudaMemcpyHostToDevice));
#else
		memcpy(m_fw_output, in, m_fw_buff_size *  sizeof(GTYPE) );
#endif
	}	

	void set_input(char* buf, int type)
	{
		if (type == sizeof(GTYPE))
		{
			memcpy(m_in, buf, m_fw_buff_size * sizeof(GTYPE));
			set_input(m_in);
			return;
		}

		switch (type)
		{
		case 1:
		{
			unsigned char value;
			for (int i = 0; i < m_fw_buff_size; i++)
			{
				memcpy(&value, buf + i * sizeof(char), sizeof(char));
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
		set_input(m_in);
	}


	void set_data_augm(bool value)
	{
		m_data_augm = value;
	}

	void set_input_type(int type) { m_input_type = type; }
	int  m_input_type = sizeof(GTYPE);

	GTYPE *m_in = nullptr;
private:
	bool m_data_augm = false;  //���ݶ����Կ���
	

#define  X_MOVE 4 
#define  Y_MOVE 4 

#define  ROT_DEGREE 30
#define  TWIST_ROW 10
#define  TWIST_COL 10

	void data_augm(GTYPE *in)
	{
		for (int i = 0; i < m_batch_num; i++)
		{
			int rot = rand() % 15;			
			switch (rot)
			{
			case 0:
				//ˮƽ��ת
				hor_rot(in + i*m_fw_batch_size, m_fw_row, m_fw_col, m_fw_node_num);
				break;
			/*case 1:
				//��ֱ��ת
				ver_rot(in + i*m_fw_batch_size, m_fw_row, m_fw_col, m_fw_node_num);
				break;*/
			case 2:	//ƽ��
			{
				int move_x = rand() % (X_MOVE * 2 + 1) - X_MOVE;
				int move_y = rand() % (Y_MOVE * 2 + 1) - Y_MOVE;
				move(in + i*m_fw_batch_size, m_fw_row, m_fw_col, m_fw_node_num, move_x, move_y);
			}
			break;
			case 3:	//С��30�ȵ���ת
			{
				int rot_degree = rand() % (ROT_DEGREE * 2 + 1) - ROT_DEGREE;
				rotate(in + i*m_fw_batch_size, m_fw_row, m_fw_col, m_fw_node_num, rot_degree);
			}
			break;
			case 4:	//Ť��
			{
				twist_data(in + i*m_fw_batch_size, m_fw_row, m_fw_col, m_fw_node_num, 0, 0, TWIST_ROW, TWIST_COL, m_fw_row, m_fw_col);
			}
			break;
			default:
				//���������ԭ��̬����
				break;
			}
		}
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
};

#endif
