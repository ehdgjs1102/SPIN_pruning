import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table  # EDIT: see deprecation warnings below
import seaborn as sns



number_blocks = [6,12,24,16]
rank_importance = [[], [], [], [], []]
pre = 'rank_densenet0.35+re_no_retrain/densenet121_limit3'

y_max = 56
rank = pre + '/rank_conv%d'%(1) + '.npy'
data = np.load(rank)
mean_rank = data.mean()
importance = mean_rank / y_max
rank_importance[0].append(importance)

cnt=1

for i in range(4):
    for j in range(number_blocks[i]):
        cnt += 1
        rank = pre + '/rank_conv%d'%(cnt) + '.npy'
        data = np.load(rank)
        mean_rank = data.mean()
        importance = mean_rank / y_max
        rank_importance[i+1].append(importance)

        cnt += 1
        rank = pre + '/rank_conv%d'%(cnt) + '.npy'
        data = np.load(rank)
        mean_rank = data.mean()
        importance = mean_rank / y_max
        rank_importance[i+1].append(importance)

      

    y_max = y_max // 2
    if i != 3:
        cnt += 1
        rank = pre + '/rank_conv%d'%(cnt) + '.npy'
        data = np.load(rank)
        mean_rank = data.mean()
        importance = mean_rank / y_max
        rank_importance[i+1].append(importance)



print(rank_importance)

df = pd.DataFrame(rank_importance)

# plt.pcolor(df)

# plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)

# plt.yticks(np.arange(0.5, len(df.index), 1), df.index)

# plt.title('mean_rank / full_rank', fontsize=14)

# plt.xlabel('conv', fontsize=10)

# plt.ylabel('layer', fontsize=10)

# plt.colorbar()
plt.figure(figsize=(32,8))
ax = sns.heatmap(df, vmin=0.2, vmax=1, annot=True, cmap='YlGnBu')

plt.title('densenet0.35+re no retrain rank importance', fontsize=30)
plt.savefig('densenet0.35+re_no_retrain_rank_importance_blue.png')
plt.clf()
ax = sns.heatmap(df, vmin=0.2, vmax=1, annot=True)
plt.title('densenet0.35+re no retrain rank importance', fontsize=30)
plt.savefig('densenet0.35+re_no_retrain_rank_importance_red.png')


# flights = sns.load_dataset("flights")
# flights = flights.pivot("month", "year", "passengers")
# plt.figure(figsize=(10, 10))
# ax = sns.heatmap(flights, annot=True, fmt="d")
# plt.savefig('rank_importance.png')


# plt.plot(rank_importance[0] , 'ro')
# plt.title( 'rank_importance0' ) 
# plt.ylim([-0.1, 1.1])
# plt.xlabel('conv') # x축 label : 'Students_num'
# plt.ylabel('mean_rank / full_rank') # y축 label : 'Score'
# plt.savefig('rank_importance0' )
# plt.close()

# plt.plot(rank_importance[1] , 'ro')
# plt.title( 'rank_importance1' ) 
# plt.ylim([-0.1, 1.1])
# plt.xlabel('conv') # x축 label : 'Students_num'
# plt.ylabel('mean_rank / full_rank') # y축 label : 'Score'
# plt.savefig('rank_importance1' )
# plt.close()

# plt.plot(rank_importance[2] , 'ro')
# plt.title( 'rank_importance2' ) 
# plt.ylim([-0.1, 1.1])
# plt.xlabel('conv') # x축 label : 'Students_num'
# plt.ylabel('mean_rank / full_rank') # y축 label : 'Score'
# plt.savefig('rank_importance2' )
# plt.close()

# plt.plot(rank_importance[3] , 'ro')
# plt.title( 'rank_importance3' ) 
# plt.ylim([-0.1, 1.1])
# plt.xlabel('conv') # x축 label : 'Students_num'
# plt.ylabel('mean_rank / full_rank') # y축 label : 'Score'
# plt.savefig('rank_importance3' )
# plt.close()

# plt.plot(rank_importance[4] , 'ro')
# plt.title( 'rank_importance5' ) 
# plt.ylim([-0.1, 1.1])
# plt.xlabel('conv') # x축 label : 'Students_num'
# plt.ylabel('mean_rank / full_rank') # y축 label : 'Score'
# plt.savefig('rank_importance5' )
# plt.close()