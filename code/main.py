# git测试cyy

import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)  #设置seed是2020
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset  #加载所选择的数据集

#执行所选择的模型
Recmodel = register.MODELS[world.model_name](world.config, dataset)   #运行model文件中所选择的模型
Recmodel = Recmodel.to(world.device)  #将模型 Recmodel 移动到指定的设备上进行计算
bpr = utils.BPRLoss(Recmodel, world.config)

# 指定的权重文件路径
weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
# 尝试从指定的文件 weight_file 中加载模型的权重
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard  指定日志的保存路径和名称
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

# 一个训练循环，其中包含了一些测试和保存模型的操作
try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %10 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()