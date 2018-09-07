#pragma once
#ifndef _TEST__H
#define _TEST__H


static int  findmax_pos(GTYPE *in, int len)
{
	int max = 0;
	GTYPE maxvalue, temp;

	maxvalue = fabs(in[0]);
	for (unsigned int i = 1; i < len; i++)
	{
		temp = fabs(in[i]);
		if (temp > maxvalue)
		{
			max = i;
			maxvalue = temp;
		}
	}
	return max;
}


/*
用于14类肺病的测试代码
*/

struct  lung_t
{
	int total_sta[14];
	int right_sta[14];
	int total_num;
	int right_num;
	GTYPE total_loss;
	bool  havelabel;
};
lung_t g_lung_data;

static void lung_init_test(Clayer *start, Clayer *end)
{
	memset(&g_lung_data, 0, sizeof(lung_t));
}

static GTYPE  lung_test(Clayer *start, Clayer *end, GTYPE *label)
{
	GTYPE *ans = new GTYPE[end->m_fw_buff_size];
	GTYPE  temp = 0;

	end->m_prev_layer[0]->get_fw_output(ans);
	if (label)
	{
		g_lung_data.havelabel = true;
		temp = dynamic_cast<End*>(end)->get_cost();
		g_lung_data.total_loss = temp;
		if (g_config.m_test_log)
			printf("\n %d cost = %f \n  ", g_lung_data.total_num, temp);
		for (int size = 0; size < end->m_batch_num; size++)
		{
			for (int i = 0; i < end->m_fw_batch_size; i++)
			{
				if (g_config.m_test_log)
					printf("(%5f, %5f), ", label[size *end->m_fw_batch_size + i], ans[size *end->m_fw_batch_size + i]);
				g_lung_data.total_num++;
				if (label[size *end->m_fw_batch_size + i] > 0.9999)
				{
					g_lung_data.total_sta[i]++;
					if (ans[size *end->m_fw_batch_size + i] > 0.50)
					{
						g_lung_data.right_sta[i]++;
						g_lung_data.right_num++;
					}
				}
				else
				{
					if (ans[size *end->m_fw_batch_size + i] < 0.50)
						g_lung_data.right_num++;
				}
			}
			printf("\n");
		}
	}
	else
	{
		g_lung_data.havelabel = false;
		for (int size = 0; size < end->m_batch_num; size++)
		{
			for (int i = 0; i < end->m_fw_batch_size; i++)
			{
				if (g_config.m_test_log)
					printf("( %5f ), ", ans[size *end->m_fw_batch_size + i]);
			}
			if (g_config.m_test_log) printf("\n");
		}
	}
	delete[] ans;
	return temp;
}

static GTYPE lung_show_test_answer(Clayer *start, Clayer *end)
{
	if (g_lung_data.havelabel)
	{
		printf("total is  %d, the right is  %d\n", g_lung_data.total_num, g_lung_data.right_num);
		for (int i = 0; i < end->m_fw_batch_size; i++)
		{
			if (g_lung_data.total_sta[i] != 0)
				printf("%d: total= %d, checkout = %d ,right rate = %f\n", i, g_lung_data.total_sta[i], g_lung_data.right_sta[i], 1.0*g_lung_data.right_sta[i] / g_lung_data.total_sta[i]);
			else
				printf("%d: total= %d, checkout = %d ,right rate = %f\n", i, g_lung_data.total_sta[i], g_lung_data.right_sta[i], 0.0);

		}
		return 1.0 - (g_lung_data.right_num *1.0) / g_lung_data.total_num;
	}
	else
	{		
	}
	return 0;
}

void  init_test(Clayer *start, Clayer *end)
{
	lung_init_test(start, end);
}

GTYPE  test_proc(Clayer *start, Clayer *end, GTYPE *label = nullptr)
{
	return lung_test(start, end, label);
}

GTYPE show_test_answer(Clayer *start, Clayer *end)
{
	return lung_show_test_answer(start, end);
}
#endif
