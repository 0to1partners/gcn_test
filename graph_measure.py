

#%%
import numpy as np
a = np.array([[1,3],[3,4]])
b = np.array([[2,6],[6,8]])

#%%
c = np.linalg.svd(a)[1]
c = np.linalg.norm(c, ord = 2)

#%%
from scipy.special import kl_div
from scipy.spatial.distance import euclidean, jensenshannon

#%%


def distance_singular(mat1, mat2):
    m1 = mat1 / mat1.sum(axis = 0)
    s1 = np.linalg.norm(np.linalg.svd(m1)[1], ord = 2)

    m2 = mat2 / mat2.sum(axis = 0)
    s2 = np.linalg.norm(np.linalg.svd(m2)[1], ord = 2)
    return np.abs(s1 - s2)


#%%

def js_matrix(mat1, mat2):
    out = jensenshannon(mat1, mat2, axis=0) + jensenshannon(mat1, mat2, axis=1)
    return out.sum()


# %%

# %%
import numpy as np
a = np.array([[1,3],[3,4]])
b = np.array([[3,4],[2,3]])
jensenshannon(a,b).sum() + jensenshannon(a,b, axis=1).sum()

# %%

a = np.array([[10,30],[30,40]])
b = np.array([[20,60],[60,80]])
jensenshannon(a,b).sum()
# %%
js_matrix(a,b)
# %%

tmp = []

tmp.append( seoul_data[main_col+age_10_col]) # 10대
tmp.append( seoul_data[main_col+age_20_col]) # 20대
tmp.append( seoul_data[main_col+age_30_col]) # 30대
tmp.append( seoul_data[main_col+age_40_col]) # 40대
tmp.append( seoul_data[main_col+age_50_col]) # 50대
tmp.append( seoul_data[main_col+age_60_col]) # 60대
tmp.append( seoul_data[main_col+age_70_col]) # 70대


out_dict = {}
for i in range(len(tmp)):
    
    tmp2 = []
    for j in range(len(tmp)):
        tmp2.append(calc_distance(tmp[i], tmp[j]))

    out_dict[i] = tmp2

data = pd.DataFrame(out_dict)#, 
data.index = ['10대', '20대', '30대', '40대', '50대', '60대', '70대']
data.columns = ['10대', '20대', '30대', '40대', '50대', '60대', '70대']
    