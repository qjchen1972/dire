#pragma once
#ifndef _NET__H
#define _NET__H

#include "dire_api/layer.h"
#include "dire_api/active.h"
#include "dire_api/batchnorm.h"
#include "dire_api/conv.h"
#include "dire_api/samp.h"
#include "dire_api/end.h"
#include "dire_api/scale.h"
#include "dire_api/start.h"
#include "dire_api/grating.h"
#include "dire_api/api_proc.h"
#include "dire_api/move.h"
#include "dire_api/limit.h"
#include "dire_api/linearmapping.h"


#define  GAN_ROW  64
#define  GAN_COL  64
#define  GAN_END_NODE  5
#define  GAN_NODE  32

void gan_net(Clayer **s, Clayer **e)
{
	// 开始构建网络	
	Start *start = new Start();
	start->set_layer_num(0);
	start->set_node_num(1);
	start->set_fw_out(GAN_ROW, GAN_COL);
	start->set_data_augm(true);
	*s = start;

	End *end = new End();
	end->set_fw_out(1, 1);
	end->set_node_num(GAN_END_NODE);
	end->set_cost_type(WEIGHT_CROSS_ENTROPY);
	//end设置权值
	GTYPE end_weight[] = { 0.515, 0.409,0.066,0.008,0.002 };
	end->set_label_weight(end_weight, sizeof(end_weight) / sizeof(GTYPE));
	*e = end;


	Conv *conv;
	Batchnorm *bn;
	Scale *scale;
	Active *active;
	Samp  *samp;
	//Grating *grating;
	Move *move;
	Limit *limit;
	LinearMapping *line;

	//跳转记录
	Clayer *jump[32];

	int layer_num = start->m_layer_num;
	int initnum = 24;
	int growrate = 12;

	conv = new Conv();
	conv->set_kernel_param(3, 3, 1, 1);
	conv->set_fw_out(GAN_ROW, GAN_COL);
	conv->set_node_num(initnum);
	conv->close_bias();
	add_net(start, conv, &layer_num);	

	//first
	//dense_block	
	Clayer *temp = conv;
	for (int i = 0; i < 6; i++)
	{
		//dense_block
		bn = new Batchnorm();
		add_net(temp, bn, &layer_num);

		scale = new Scale();
		add_net(bn, scale, &layer_num);

		active = new Active();
		active->set_atcive_type(RELU);
		add_net(scale, active, &layer_num);

		jump[i] = active;

		conv = new Conv();
		conv->set_kernel_param(1, 1, 1, 1);
		conv->set_fw_out(GAN_ROW, GAN_COL);
		conv->set_node_num(4 * growrate);
		conv->close_bias();
		add_net(active, conv, &layer_num);
		for (int k = 0; k < i; k++)
			add_net(jump[k], conv);

		bn = new Batchnorm();
		add_net(conv, bn, &layer_num);

		scale = new Scale();
		add_net(bn, scale, &layer_num);

		active = new Active();
		active->set_atcive_type(RELU);
		add_net(scale, active, &layer_num);

		conv = new Conv();
		conv->set_kernel_param(3, 3, 1, 1);
		conv->set_fw_out(GAN_ROW, GAN_COL);
		conv->set_node_num(growrate);
		conv->close_bias();
		add_net(active, conv, &layer_num);

		line = new LinearMapping();
		line->set_node_num(growrate);
		add_net(conv, line, &layer_num);

		temp = line;
		//end
	}

	// second
	bn = new Batchnorm();
	add_net(temp, bn, &layer_num);

	scale = new Scale();
	add_net(bn, scale, &layer_num);

	active = new Active();
	active->set_atcive_type(RELU);
	add_net(scale, active, &layer_num);

	initnum = (initnum + 6*growrate)/2;

	conv = new Conv();
	conv->set_kernel_param(1, 1, 1, 1);
	conv->set_fw_out(GAN_ROW, GAN_COL);
	conv->set_node_num(initnum);
	conv->close_bias();
	add_net(active, conv, &layer_num);
	for (int k = 0; k < 6; k++)
		add_net(jump[k], conv);
	
	/*samp = new Samp();
	samp->set_param(AVG_POOL, 2, 2, 2, 2);
	samp->close_bias();
	add_net(conv, samp, &layer_num);
	*/

	temp = conv;
	conv = new Conv();
	conv->set_kernel_param(3, 1, 2, 2);
	conv->set_fw_out(GAN_ROW/2, GAN_COL/2);
	conv->close_bias();
	add_net(temp, conv, &layer_num);

	temp = conv;
	for (int i = 0; i < 6; i++)
	{
		//dense_block
		bn = new Batchnorm();
		add_net(temp, bn, &layer_num);

		scale = new Scale();
		add_net(bn, scale, &layer_num);

		active = new Active();
		active->set_atcive_type(RELU);
		add_net(scale, active, &layer_num);

		jump[i] = active;

		conv = new Conv();
		conv->set_kernel_param(1, 1, 1, 1);
		conv->set_fw_out(GAN_ROW / 2, GAN_COL / 2);
		conv->set_node_num(4* growrate);
		conv->close_bias();
		add_net(active, conv, &layer_num);
		for (int k = 0; k < i; k++)
			add_net(jump[k], conv);

		bn = new Batchnorm();
		add_net(conv, bn, &layer_num);
		
		scale = new Scale();
		add_net(bn, scale, &layer_num);

		active = new Active();
		active->set_atcive_type(RELU);
		add_net(scale, active, &layer_num);

		conv = new Conv();
		conv->set_kernel_param(3, 3, 1, 1);
		conv->set_fw_out(GAN_ROW / 2, GAN_COL / 2);
		conv->set_node_num(growrate);
		conv->close_bias();
		add_net(active, conv, &layer_num);

		line = new LinearMapping();
		line->set_node_num(growrate);
		add_net(conv, line, &layer_num);

		temp = line;
		//end
	}

	bn = new Batchnorm();
	add_net(temp, bn, &layer_num);

	scale = new Scale();
	add_net(bn, scale, &layer_num);

	active = new Active();
	active->set_atcive_type(RELU);
	add_net(scale, active, &layer_num);

	initnum = (initnum + 6 * growrate) / 2;

	conv = new Conv();
	conv->set_kernel_param(1, 1, 1, 1);
	conv->set_fw_out(GAN_ROW/2, GAN_COL/2);
	conv->set_node_num(initnum);
	conv->close_bias();
	add_net(active, conv, &layer_num);
	for (int k = 0; k < 6; k++)
		add_net(jump[k], conv);

	temp = conv;
	conv = new Conv();
	conv->set_kernel_param(4, 1, 2, 2);
	conv->set_fw_out(GAN_ROW / 4, GAN_COL / 4);
	conv->close_bias();
	add_net(temp, conv, &layer_num);

	temp = conv;

	/*samp = new Samp();
	samp->set_param(AVG_POOL, 2, 2, 2, 2);
	samp->close_bias();
	add_net(conv, samp, &layer_num);
	
	temp = samp;
	*/

	for (int i = 0; i < 6; i++)
	{
		//dense_block
		bn = new Batchnorm();
		add_net(temp, bn, &layer_num);

		scale = new Scale();
		add_net(bn, scale, &layer_num);

		active = new Active();
		active->set_atcive_type(RELU);
		add_net(scale, active, &layer_num);

		jump[i] = active;

		conv = new Conv();
		conv->set_kernel_param(1, 1, 1, 1);
		conv->set_fw_out(GAN_ROW / 4, GAN_COL / 4);
		conv->set_node_num(4* growrate);
		conv->close_bias();
		add_net(active, conv, &layer_num);
		for (int k = 0; k < i; k++)
			add_net(jump[k], conv);


		bn = new Batchnorm();
		add_net(conv, bn, &layer_num);

	
		scale = new Scale();
		add_net(bn, scale, &layer_num);

		active = new Active();
		active->set_atcive_type(RELU);
		add_net(scale, active, &layer_num);

		conv = new Conv();
		conv->set_kernel_param(3, 3, 1, 1);
		conv->set_fw_out(GAN_ROW / 4, GAN_COL / 4);
		conv->set_node_num(growrate);
		conv->close_bias();
		add_net(active, conv, &layer_num);

		line = new LinearMapping();
		line->set_node_num(growrate);
		add_net(conv, line, &layer_num);
		temp = line;
		//end
	}
	
	bn = new Batchnorm();
	add_net(temp, bn, &layer_num);

	scale = new Scale();
	add_net(bn, scale, &layer_num);

	active = new Active();
	active->set_atcive_type(RELU);
	add_net(scale, active, &layer_num);

	initnum = (initnum + 6 * growrate) / 2;

	conv = new Conv();
	conv->set_kernel_param(1, 1, 1, 1);
	conv->set_fw_out(GAN_ROW / 4, GAN_COL / 4);
	conv->set_node_num(initnum);
	add_net(active, conv, &layer_num);
	for (int k = 0; k < 6; k++)
		add_net(jump[k], conv);

	temp = conv;
	conv = new Conv();
	conv->set_kernel_param(5, 1, 2, 2);
	conv->set_fw_out(GAN_ROW / 8, GAN_COL / 8);
	conv->close_bias();
	add_net(temp, conv, &layer_num);

	temp = conv;

	/*samp = new Samp();
	samp->set_param(AVG_POOL, 2, 2, 2, 2);
	samp->close_bias();
	add_net(conv, samp, &layer_num);

	temp = samp;
	*/
	for (int i = 0; i < 6; i++)
	{
		//dense_block
		bn = new Batchnorm();
		add_net(temp, bn, &layer_num);

		scale = new Scale();
		add_net(bn, scale, &layer_num);

		active = new Active();
		active->set_atcive_type(RELU);
		add_net(scale, active, &layer_num);

		jump[i] = active;

		conv = new Conv();
		conv->set_kernel_param(1, 1, 1, 1);
		conv->set_fw_out(GAN_ROW / 8, GAN_COL / 8);
		conv->set_node_num(growrate * 4);
		conv->close_bias();
		add_net(active, conv, &layer_num);
		for (int k = 0; k < i; k++)
			add_net(jump[k], conv);

	
		bn = new Batchnorm();
		add_net(conv, bn, &layer_num);		

		scale = new Scale();
		add_net(bn, scale, &layer_num);

		active = new Active();
		active->set_atcive_type(RELU);
		add_net(scale, active, &layer_num);

		conv = new Conv();
		conv->set_kernel_param(3, 3, 1, 1);
		conv->set_fw_out(GAN_ROW / 8, GAN_COL / 8);
		conv->set_node_num(growrate);
		conv->close_bias();
		add_net(active, conv, &layer_num);

		line = new LinearMapping();
		line->set_node_num(growrate);
		add_net(conv, line, &layer_num);
		temp = line;
		//end
	}

	bn = new Batchnorm();
	add_net(temp, bn, &layer_num);

	scale = new Scale();
	add_net(bn, scale, &layer_num);

	active = new Active();
	active->set_atcive_type(RELU);
	add_net(scale, active, &layer_num);

	samp = new Samp();
	samp->set_param(GLOBAL_AVG_POOL, GAN_ROW / 8, GAN_COL / 8, 1, 1);
	add_net(active, samp, &layer_num);
	for (int k = 0; k < 6; k++)
		add_net(jump[k], samp);	

	conv = new Conv();
	conv->set_node_num(end->m_fw_node_num);
	add_net(samp, conv, &layer_num);

    move = new Move();
	add_net(conv, move, &layer_num);

	active = new Active();
	active->set_atcive_type(SOFTMAX);
	add_net(move, active, &layer_num);

	limit = new Limit();
	add_net(active, limit, &layer_num);	
	limit->fw_need_write_file = true;

	add_net(limit, end, &layer_num);
}

void build_net(Clayer **start, Clayer **end)
{
	gan_net(start, end);
}
#endif
