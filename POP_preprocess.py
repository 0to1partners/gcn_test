#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import font_manager, rc
rc('font', family='NanumGothic')
import pickle
#%%
data2= pd.read_csv('data/LOCAL_PEOPLE_GU_2021.csv', encoding='cp949')
data2['TIME'] = data2.apply(lambda x : f"{str(int(x['기준일ID']))}T{int(x['시간대구분']):02d}", axis=1)
data2 = data2.drop(['기준일ID', '시간대구분','총생활인구수'], axis=1)

data2 = data2.rename({
    '자치구코드':'SGG_CD',
    '남자0세부터9세생활인구수' : 'M00',
    '남자10세부터14세생활인구수' : 'M10',
    '남자15세부터19세생활인구수' : 'M15',
    '남자20세부터24세생활인구수' : 'M20',
    '남자25세부터29세생활인구수' : 'M25',
    '남자30세부터34세생활인구수' : 'M30',
    '남자35세부터39세생활인구수' : 'M35',
    '남자40세부터44세생활인구수' : 'M40',
    '남자45세부터49세생활인구수' : 'M45',
    '남자50세부터54세생활인구수' : 'M50',
    '남자55세부터59세생활인구수' : 'M55',
    '남자60세부터64세생활인구수' : 'M60',
    '남자65세부터69세생활인구수' : 'M65',
    '남자70세이상생활인구수' : 'M70',
    '여자0세부터9세생활인구수' : 'F00',
    '여자10세부터14세생활인구수' : 'F10',
    '여자15세부터19세생활인구수' : 'F15',
    '여자20세부터24세생활인구수' : 'F20',
    '여자25세부터29세생활인구수' : 'F25',
    '여자30세부터34세생활인구수' : 'F30',
    '여자35세부터39세생활인구수' : 'F35',
    '여자40세부터44세생활인구수' : 'F40',
    '여자45세부터49세생활인구수' : 'F45',
    '여자50세부터54세생활인구수' : 'F50',
    '여자55세부터59세생활인구수' : 'F55',
    '여자60세부터64세생활인구수' : 'F60',
    '여자65세부터69세생활인구수' : 'F65',
    '여자70세이상생활인구수' : 'F70',
}, axis=1)

#%%
from tqdm.auto import tqdm
tqdm.pandas()

data_list = []
for i in tqdm(data2['TIME'].unique()):
    tmp = data2[data2['TIME'] == i].drop(['TIME'], axis=1) 
    tmp = tmp.set_index('SGG_CD')
    data_list.append(tmp )
#%%
data = np.array(data_list)
data.shape

#%%
# numpy.save 
np.save('data/LOCAL_PEOPLE_GU_2021.npy', data)



