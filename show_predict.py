#!/usr/bin/python
# -*- coding: Shift_JIS -*-

#######################################################################
# jupyter-notebook上で動かす。真の値と、予測値をグラフ化します。
# このスクリプトの実行の前に、predict.pyを実行すること。

#######################################################################
## list 16
# インポート＆実行済みの場合、以下の3行はなくてもよい
import pandas as pd
import cufflinks as cf
cf.go_offline()

# 正解データと予測データ
correct_2005_year = dataset[dataset.times.year >= 2005].series_data
predict_2005_year = predict_air_quality

# 2005年3月分のデータに絞るには、コメントアウトを外す
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
    
    
    
    
    
    
    
    
    