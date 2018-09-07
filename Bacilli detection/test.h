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
   用于杆菌测试的代码
*/

struct  gan_t
{
	int right_sta[5];
	int err[5][5];
	int right_num;
	int total_num;
	GTYPE total_loss;
	bool  havelabel;
};
gan_t g_gan_data;

static void gan_init_test(Clayer *start, Clayer *end)
{
	memset(&g_gan_data, 0, sizeof(gan_t));
}

static GTYPE  gan_test(Clayer *start, Clayer *end, GTYPE *label)
{
	GTYPE *ans = new GTYPE[end->m_fw_buff_size];
	GTYPE  temp = 0;

	end->m_prev_layer[0]->get_fw_output(ans);

	if (label)
	{
		g_gan_data.havelabel = true;
		temp = dynamic_cast<End*>(end)->get_cost();
		g_gan_data.total_loss = temp;
		if(g_config.m_test_log)
			printf("\n %d cost = %f \n  ", g_gan_data.total_num, temp);
		for (int size = 0; size < end->m_batch_num; size++)
		{
			int label_max = findmax_pos(label + size *end->m_fw_batch_size, end->m_fw_batch_size);
			g_gan_data.right_sta[label_max]++;
			if (g_config.m_test_log)
			{
				for (int i = 0; i < end->m_fw_batch_size; i++)
					printf("(%5f, %5f), ", label[size *end->m_fw_batch_size + i], ans[size *end->m_fw_batch_size + i]);
				printf("\n");
			}
			int ans_max = findmax_pos(ans + size *end->m_fw_batch_size, end->m_fw_batch_size);
			if (label_max == ans_max) g_gan_data.right_num++;
			g_gan_data.err[label_max][ans_max]++;
			g_gan_data.total_num++;
		}
	}
	else
	{
		g_gan_data.havelabel = false;
		for (int size = 0; size < end->m_batch_num; size++)
		{
			int ans_max = findmax_pos(ans + size *end->m_fw_batch_size, end->m_fw_batch_size);
			g_gan_data.err[ans_max][ans_max]++;
			g_gan_data.total_num++;
		}
	}
	delete[] ans;
	return temp;
}

static GTYPE gan_show_test_answer(Clayer *start, Clayer *end)
{
	if (g_gan_data.havelabel)
	{
		printf("total is  %d, the right is  %d\n", g_gan_data.total_num, g_gan_data.right_num);
		for (int i = 0; i < end->m_fw_batch_size; i++)
		{
			if(g_gan_data.right_sta[i] != 0 )
				printf("%d: total= %d, checkout = %d ,right rate = %f\n", i, g_gan_data.right_sta[i], g_gan_data.err[i][i], 1.0*g_gan_data.err[i][i] / g_gan_data.right_sta[i]);
			else
				printf("%d: total= %d, checkout = %d ,right rate = %f\n", i, g_gan_data.right_sta[i], g_gan_data.err[i][i], 0.0);

		}
		for (int i = 0; i < 5; i++)
			printf("%10d    %10d    %10d    %10d    %10d\n", g_gan_data.err[i][0], g_gan_data.err[i][1], g_gan_data.err[i][2], g_gan_data.err[i][3], g_gan_data.err[i][4]);
		return 1.0 - (g_gan_data.right_num *1.0) / g_gan_data.total_num;
	}
	else
	{
		if (g_config.m_test_log)
		{
			for (int i = 0; i < end->m_fw_batch_size; i++)
				printf("%d: checkout = %d \n", i, g_gan_data.err[i][i]);
		}		
	}
	return 0;
}


void  init_test(Clayer *start, Clayer *end)
{
	gan_init_test(start, end);
}

GTYPE  test_proc(Clayer *start, Clayer *end, GTYPE *label = nullptr)
{
	return gan_test(start, end, label);
}

GTYPE show_test_answer(Clayer *start, Clayer *end)
{
	return gan_show_test_answer(start, end);
}
#endif
