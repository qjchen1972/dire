/*
* Copyright (c) 2017
* All rights reserved.
*
* �ļ�����: gpu.h
* �ļ���ʶ:
* ժ    Ҫ: gpu�����һЩ��������
*
*
* ��ǰ�汾: 1.0
* ��    ��: chen.qian.jiang
* ��ʼ����: 2017-07-15
* �������:
* ����˵����
* �޸�����        �汾��     �޸���          �޸�����
* -----------------------------------------------
* 2017/07/15      V1.0       ��ǭ��          ����
* 2017/08/01      V1.0       ��ǭ��          �޸��Ż�
* 2017/08/11                                 ȥ��ͳһѰַ
*/
#pragma once

#ifndef _GPU__H
#define _GPU__H

#include "const.h"
#include "proc_gpu.h"

#ifdef GPU


class Gpu
{
public:
	Gpu(){}
	~Gpu(){}		 
	//����������
	void gpu_cal(struct gpu_t st, int fn_type)
	{
		int  block;
		dim3  gird;
		unsigned int count;

		get_gpu_dim(st.len, gird, block,count);
		cal <<<gird, block >>> (st,fn_type);
		cudaDeviceSynchronize();
	}

	//������ۼƼ���,Ŀǰֻ֧��gird<1024,���Ǵ���1024��������������á�
	//2017.11.14�޸�Ϊ֧�����ⷶΧ���������	
	void gpu_sum(gpu_t st, int fn_type)
	{
		int block;
		dim3 gird;

		unsigned int  blknum;
		unsigned int xgird;

		get_gpu_dim(st.len, gird, block, blknum);
		
		if (blknum > 1)
		{
			xgird = blknum * st.sum_result_count;
			get_gird(xgird, gird);
			st.istemp = 1;
			st.per_block_result = m_first_buf;
			block_sum << <gird, block >> > (st, fn_type);
			cudaDeviceSynchronize();
		
			st.len = blknum;
			get_gpu_dim(st.len, gird, block, blknum);
            int count = 0;
			while (blknum > 1)
			{
				xgird = blknum * st.sum_result_count;
			    get_gird(xgird, gird);
				st.istemp = 1;
				if (count % 2 == 0)
					st.result = m_first_buf;
				else
					st.result = m_second_buf;
				st.start = 0;
				if (count % 2 == 0)
					st.per_block_result = m_second_buf;
				else
					st.per_block_result = m_first_buf;
				block_sum << <gird, block >> > (st, FN_GPU_SUM);
				cudaDeviceSynchronize();
		
				count++;

				st.len = blknum;
				get_gpu_dim(st.len, gird, block, blknum);
			}

			xgird = blknum * st.sum_result_count;
			get_gird(xgird, gird);
			if(count % 2 == 0)
				st.result = m_first_buf;
			else
				st.result = m_second_buf;
			st.start = 0;
			st.istemp = 0;
			block_sum << <gird, block >> > (st, FN_GPU_SUM);
			cudaDeviceSynchronize();
		}
		else
		{
			xgird = blknum * st.sum_result_count;
			get_gird(xgird, gird);
			block_sum << <gird, block >> > (st, fn_type);
			cudaDeviceSynchronize();
		}
	}

	void set_global_mem()
	{
		if(!m_first_buf) HANDLE_ERROR(cudaMalloc((void **)&m_first_buf, sizeof(GTYPE)*MAX_SUM_MEM));
		if (!m_second_buf) HANDLE_ERROR(cudaMalloc((void **)&m_second_buf, sizeof(GTYPE)*MAX_SUM_MEM/8));
		if(!m_in_buf) HANDLE_ERROR(cudaMalloc((void **)&m_in_buf, sizeof(unsigned long long)*MAX_CONN_NUM));
		if (!m_in_layer_buf) HANDLE_ERROR(cudaMalloc((void **)&m_in_layer_buf, sizeof(unsigned int)*MAX_CONN_NUM));
		if (!m_in_pos_buf) HANDLE_ERROR(cudaMalloc((void **)&m_in_pos_buf, sizeof(unsigned int)*MAX_CONN_NUM));
	}

	gpu_layer_t   m_cpu_layer;
	static GTYPE* m_first_buf;
	static GTYPE* m_second_buf;
	static unsigned long long* m_in_buf;
	static unsigned int* m_in_layer_buf;
	static unsigned int* m_in_pos_buf;

private:	
	void get_gpu_dim(unsigned int len_in, dim3 &gird, int &block, unsigned int &blknum )
	{
		if (len_in <= GPU_BLOCK)
		{
			block = len_in;
			gird.x = 1;
			gird.y = 1;
			blknum = 1;
		}
		else
		{
			//len_in > 0
			block = GPU_BLOCK;
			
			unsigned int xgird = len_in / GPU_BLOCK;
			if (len_in % GPU_BLOCK != 0) xgird += 1;
			blknum = xgird;
			get_gird(xgird, gird);			
		}
	}

	void get_gird(unsigned int blknum, dim3 &gird)
	{
		if (blknum <= GPU_GRID)
		{
			gird.x = blknum;
			gird.y = 1;
		}
		else
		{
			// so xgird > 0
			gird.y = blknum / GPU_GRID;
			if (blknum % GPU_GRID != 0) gird.y += 1;
			gird.x = GPU_GRID;
		}
	}
};

GTYPE*  Gpu::m_first_buf = nullptr;
GTYPE*  Gpu::m_second_buf = nullptr;
unsigned long long* Gpu::m_in_buf = nullptr;
unsigned int* Gpu::m_in_layer_buf = nullptr;
unsigned int* Gpu::m_in_pos_buf = nullptr;
#endif // GPU
#endif // _GPU__H
