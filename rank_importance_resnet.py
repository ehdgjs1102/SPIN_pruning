import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table  # EDIT: see deprecation warnings below
import seaborn as sns

prefix = 'rank_resnet0.2_no_retrain/resnet_50_limit5/rank_conv'
subfix = '.npy'
pre = 'rank_resnet0.2_no_retrain'
cnt=1
rank_importance = [[], [], [], [], []]
    
y_max = 56
rank = prefix + str(cnt) + subfix
data = np.load(rank)
mean_rank = data.mean()
importance = mean_rank / y_max
rank_importance[0].append(importance)    

cnt+=1


current_cfg = [3, 4, 6, 3]
for layer, num in enumerate(current_cfg):

    for k in range(num):
        iter = 3
        if k==0:
            iter +=1

        if layer != 0 and k == 1:
            y_max = y_max // 2

        for l in range(iter):
            rank = prefix + str(cnt) + subfix  
            data = np.load(rank)
            mean_rank = data.mean()
            importance = mean_rank / y_max
            rank_importance[layer+1].append(importance)
            cnt+=1

print(rank_importance)

df = pd.DataFrame(rank_importance)
plt.figure(figsize=(32,8))
ax = sns.heatmap(df, vmin=0.2, vmax=1, annot=True, cmap='YlGnBu')

plt.title('resnet0.2 no retrain rank importance', fontsize=30)
plt.savefig('resnet0.2_no_retrain_rank_importance_blue.png')
plt.clf()
ax = sns.heatmap(df, vmin=0.2, vmax=1, annot=True)
plt.title('resnet0.2 no retrain rank importance', fontsize=30)
plt.savefig('resnet0.2_no_retrain_rank_importance_red.png')
