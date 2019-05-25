# ！/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/20 下午5:44
# @Author  : Ryu
# @Site    : 
# @File    : GetAllScores.py
# @Software: PyCharm

import numpy as np

alorithum_list = ['wgan','wgan_dc','wgan_gp','sngan','wgan_2d']

f = open (r'./result.txt','w')

for i in alorithum_list:
    score_general_path = './result/'+i+'/cifar10/test/score.npy'
    score = np.load(score_general_path)
    print(i+':'
          +' MMD-pixl:'+str(score[1])
          +' MMD-conv:'+str(score[8])
          +' MMD-logit:'+str(score[15])
          +' MMD-smax:'+str(score[22])
          +' INS:'+str(score[28])
          +' MOS:'+str(score[29])
          +' FID:'+str(score[30]),file=f)