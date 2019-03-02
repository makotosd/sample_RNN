#!/usr/bin/python
# -*- coding: Shift_JIS -*-

#######################################################################
# jupyter-notebook��œ������B�^�̒l�ƁA�\���l���O���t�����܂��B
# ���̃X�N���v�g�̎��s�̑O�ɁApredict.py�����s���邱�ƁB

#######################################################################
## list 16
# �C���|�[�g�����s�ς݂̏ꍇ�A�ȉ���3�s�͂Ȃ��Ă��悢
import pandas as pd
import cufflinks as cf
cf.go_offline()

# �����f�[�^�Ɨ\���f�[�^
correct_2005_year = dataset[dataset.times.year >= 2005].series_data
predict_2005_year = predict_air_quality

# 2005�N3�����̃f�[�^�ɍi��ɂ́A�R�����g�A�E�g���O��
dt_2005march = pd.date_range('20050301','20050401', freq="H")
correct_2005_year = correct_2005_year.reindex(dt_2005march)
predict_2005_year = predict_2005_year.reindex(dt_2005march)

for feature in air_quality.columns:
    plot_data = pd.DataFrame({
        'mesuared': correct_2005_year[feature],
        'predicted': predict_2005_year[feature]
    }).iplot(
      asFigure = True,
      title = feature
    )
    plot_data['layout']['paper_bgcolor'] = '#FFFFFF'
    plot_data.iplot()
    
    
    
    
    
    
    
    
    