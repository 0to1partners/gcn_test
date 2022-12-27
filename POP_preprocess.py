#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import font_manager, rc
rc('font', family='NanumGothic')
import pickle

#%%
{
    '서울' : {
        11110 : '종로구',11140 : '중구',11170 : '용산구',11200 : '성동구',11215 : '광진구',
        11230 : '동대문구',11260 : '중랑구',11290 : '성북구',11305 : '강북구',11320 : '도봉구',
        11350 : '노원구',11380 : '은평구',11410 : '서대문구',11440 : '마포구',11470 : '양천구',
        11500 : '강서구',11530 : '구로구',11545 : '금천구',11560 : '영등포구',11590 : '동작구',
        11620 : '관악구',11650 : '서초구',11680 : '강남구',11710 : '송파구',11740 : '강동구'}
}

#%%
data = pd.read_csv('data/LOCAL_PEOPLE_GU_2020.csv', encoding='cp949')
data2= pd.read_csv('data/LOCAL_PEOPLE_GU_2021.csv', encoding='cp949')
data2 = pd.concat([data, data2], axis=0)

#%%

data2['TIME'] = data2.apply(lambda x : f"{str(int(x['기준일ID']))}T{int(x['시간대구분']):02d}", axis=1)
data2 = data2.drop(['기준일ID', '시간대구분','총생활인구수'], axis=1)



#%%
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
    
#%%
output = np.array(data_list)
np.save('data/LOCAL_PEOPLE_GU_2020_2021.npy', output)
output2 = output.reshape(-1, 24, 25, 28)
output2 = output2.mean(axis=1)
np.save('data/LOCAL_PEOPLE_GU_DAILY_2020_2021.npy', output2)

