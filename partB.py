
import pandas as pd
import numpy as np
from numpy.linalg import inv
from numpy.random import uniform, multivariate_normal, rand, randn, seed
from itertools import repeat
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from matplotlib.colors import to_rgba
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import jarque_bera, normaltest

# preprocessing
data = pd.read_csv('Wholesale customers data.csv',header=0)
data.shape
data.info()
data.describe().append(data.nunique().to_frame('nunique').T)
# drop Channel,Region
del data['Channel']
del data['Region']

products=data.columns.tolist()

# outliers
ax = plt.figure(figsize=(12, 5)).gca(title='', 
                                     xlabel='Sales ($)', ylabel='Product')
flierprops = dict(markerfacecolor='0.75', markersize=5, linestyle='none')
whiskerprops = capprops = dict(c='white')
sns.boxplot(data=data, orient='horizontal', 
    flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops)
plt.savefig('Distribution.png',dpi=400,bbox_inches='tight')
            
scaledData = np.log(data)

ax = plt.figure(figsize=(8, 5)).gca(title='Log Sales Distribution', 
                                    xlabel='Product', ylabel='Log $')
sns.violinplot(data=scaledData)
plt.savefig('Violin.png',dpi=400,bbox_inches='tight')

# remove outliers
outliers = LocalOutlierFactor(n_neighbors=20, contamination=.05)
scaledData['inlier'] = outliers.fit_predict(scaledData)
cleanData = scaledData.loc[scaledData.inlier==1, products]
sns.pairplot(cleanData, plot_kws={'s': 5})
plt.tight_layout();

sns.clustermap(cleanData.corr(), 
               annot=True, fmt='.1%', center=0.0, 
               vmin=-1, vmax=1, cmap=sns.diverging_palette(250, 10, n=20))
plt.savefig('Heatmap.png',dpi=400,bbox_inches='tight')


# PCA with 2 components
              
pca = PCA(n_components=2,svd_solver='full')
x, y = pca.fit_transform(cleanData).T
exp_1, exp_2 = pca.explained_variance_ratio_
print('Components kept: 2\nExplained variance=',pca.explained_variance_ratio_.sum())
pca.components_.T

ax = sns.jointplot(x=x, y=y, stat_func=None, 
                   joint_kws={'s':5}).ax_joint
ax.set_xlabel('Principal Component 1 ({:.1%})'.format(exp_1), fontsize=12)
ax.set_ylabel('Principal Component 2 ({:.1%})'.format(exp_2), fontsize=12)

arrow_size, text_pos = 5, 7
for i, component in enumerate(pca.components_.T):
    ax.arrow(0, 0, arrow_size * component[0], arrow_size * component[1],
             head_width=0.2, head_length=0.2, linewidth=1, color='red')
    ax.text(component[0] * text_pos, component[1] * text_pos, products[i], 
            color='white', ha='center', va='center', fontsize=10)
plt.savefig('PCA_components.png',dpi=400,bbox_inches='tight')

# PCA 

pca = PCA()

transformed = pca.fit_transform(cleanData)
components = pd.DataFrame(pca.components_, columns=products, 
                    index=['Component {}'.format(i) for i in range(1, 7)])
ax = components.plot.bar(figsize=(12, 8), rot=0, 
                         title='Feature Weights by Principal Component')
ax.set_ylabel('Feature Weights')
for i, exp_var in enumerate(pca.explained_variance_ratio_):
    ax.text(i-.4, ax.get_ylim()[1] - .15, 
            'Explained\nVariance\n{:.2%}'.format(exp_var))
plt.legend(loc=3)
plt.tight_layout();
plt.savefig('Weights.png',dpi=400,bbox_inches='tight')










