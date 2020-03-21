#%%
import numpy as np
import pandas as pd


df0 = pd.DataFrame(data=np.zeros([3, 3]) ,columns=['A', 'B', 'C'])
df1 = pd.DataFrame(data=np.ones([3, 3]) ,columns=['A', 'B', 'C'])
df_concat = pd.concat([df0, df1])
l = [df0, df1, df_concat]

display(df0)
display(df1)
display(df_concat)
# %%
df_all = pd.DataFrame(None)
for df in l:
    df_all = pd.concat([df_all, df])

df_all


# %%
