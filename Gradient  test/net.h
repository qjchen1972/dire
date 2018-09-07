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
#define  IMG_NODE 2


void check_net(Clayer **s, Clayer **e)
{
	// 开始构建网络	
	Start *start = new Start();
	start->set_layer_num(0);
	start->set_node_num(1);
	start->set_fw_out(32, 32);
	//start->set_data_augm(true);
	*s = start;

	End *end = new End();
	end->set_fw_out(1, 1);
	end->set_node_num(10);
	end->set_cost_type(WEIGHT_CROSS_ENTROPY);
	//end设置权值
	GTYPE end_weight[] = { 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1 };
	end->set_label_weight(end_weight, sizeof(end_weight) / sizeof(GTYPE));
	*e = end;

	Conv *conv;
	Batchnorm *bn;
	Scale *scale;
	Active *active;
	Samp  *samp;
	Grating *grating;
	Move *move;
	Limit *limit;
	LinearMapping *line;

	Clayer *jump1, *jump2, *jump3;
	int layer_num = start->m_layer_num;

	conv = new Conv();
	conv->set_kernel_param(3, 3, 1, 1);
	conv->set_fw_out(start->m_fw_row, start->m_fw_col);
	conv->set_node_num(IMG_NODE);
	add_net(start, conv, &layer_num);

	line = new LinearMapping();
	line->set_node_num(2 * IMG_NODE);
	add_net(conv, line, &layer_num);

	bn = new Batchnorm();
	add_net(line, bn, &layer_num);

	scale = new Scale();
	add_net(bn, scale, &layer_num);

	active = new Active();
	active->set_atcive_type(LEAKY_RELU);
	add_net(scale, active, &layer_num);

	jump1 = active;

	conv = new Conv();
	conv->set_kernel_param(3, 3, 1, 1);
	conv->set_fw_out(start->m_fw_row, start->m_fw_col);
	conv->set_node_num(IMG_NODE);
	add_net(active, conv, &layer_num);

	line = new LinearMapping();
	line->set_node_num(2 * IMG_NODE);
	add_net(conv, line, &layer_num);

	bn = new Batchnorm();
	add_net(line, bn, &layer_num);


	scale = new Scale();
	add_net(bn, scale, &layer_num);

	active = new Active();
	active->set_atcive_type(RELU);
	add_net(scale, active, &layer_num);

	jump2 = active;

	conv = new Conv();
	conv->set_kernel_param(3, 3, 1, 1);
	conv->set_fw_out(start->m_fw_row, start->m_fw_col);
	conv->set_node_num(IMG_NODE);
	add_net(active, conv, &layer_num);
	add_net(jump1, conv);


	bn = new Batchnorm();
	add_net(conv, bn, &layer_num);

	scale = new Scale();
	add_net(bn, scale, &layer_num);

	active = new Active();
	active->set_atcive_type(RELU);
	add_net(scale, active, &layer_num);

	conv = new Conv();
	conv->set_kernel_param(3, 3, 1, 1);
	conv->set_fw_out(start->m_fw_row, start->m_fw_col);
	conv->set_node_num(IMG_NODE);
	add_net(active, conv, &layer_num);
	add_net(jump1, conv);
	add_net(jump2, conv);


	bn = new Batchnorm();
	add_net(conv, bn, &layer_num);

	scale = new Scale();
	add_net(bn, scale, &layer_num);

	active = new Active();
	active->set_atcive_type(RELU);
	add_net(scale, active, &layer_num);

	samp = new Samp();
	samp->set_param(MAX_POOL, 3, 2, 2, 3);
	add_net(active, samp, &layer_num);

	/*grating = new Grating();
	grating->set_type(GRATING_TYPE);
	add_net(samp, grating, &layer_num);

	conv = new Conv();
	conv->set_node_num(10);
	add_net(grating, conv, &layer_num);
	//jump1 = conv;

	grating = new Grating();
	grating->set_type(GRATING_TYPE);
	add_net(conv, grating, &layer_num);
	*/

	conv = new Conv();
	conv->set_node_num(end->m_fw_node_num);
	add_net(samp, conv, &layer_num);

	//bn = new Batchnorm();
	//add_net(conv, bn, &layer_num);



	move = new Move();
	add_net(conv, move, &layer_num);

	active = new Active();
	active->set_atcive_type(SOFTMAX);
	add_net(move, active, &layer_num);

	limit = new Limit();
	add_net(active, limit, &layer_num);

	add_net(limit, end, &layer_num);
}




void build_net(Clayer **start, Clayer **end)
{
	check_net(start, end);
}
#endif
