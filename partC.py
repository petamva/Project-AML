
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA



data = pd.read_csv('houseData.csv')

data.describe()

pd.isna(data).sum().sum()

#-------Data cleaning

dataReduced = data.drop(['id','date','waterfront','yr_renovated','zipcode','lat','long'],axis=1)

features = dataReduced.columns.tolist()

dataReduced.describe().append(dataReduced.nunique().to_frame('nunique').T)

dataReduced.info()

#------Correlation Heatmap
 
sns.clustermap(dataReduced.iloc[:,1:].corr(), 
               annot=True, fmt='.1%', center=0.0, 
               vmin=-1, vmax=1, cmap=sns.diverging_palette(250, 10, n=20))

#-----Normalization

dataProcessed = preprocessing.scale(dataReduced)

#-------Data spliting

target=dataProcessed[:,0]

dataFeatures = dataProcessed[:,1:]

#------Feature correlation to target
 
fig = plt.figure()

for i in range(dataFeatures.shape[1]):
    ax = fig.add_subplot(4,4,i+1)
    ax.scatter(dataFeatures[:,i], target, s=0.1, alpha = 0.5,color='black')
    plt.xlabel(features[i+1])
    plt.ylabel("Price")
    
plt.tight_layout()
#plt.savefig('test.png',dpi=400,bbox_inches='tight')
    

#------Outlier removal

threshold = 3

for i in range(dataProcessed.shape[1]):
    zscore = stats.zscore(dataProcessed[:,i])
    median = np.nanmedian(dataProcessed[:,i])
    dataProcessed[:,i] = np.where(np.abs(zscore) < threshold,dataProcessed[:,i],median)


ax = plt.figure(figsize=(12, 5)).gca(title='', 
                                     xlabel='Distribution', ylabel='Features')
flierprops = dict(markerfacecolor='0.75', markersize=5, linestyle='none')
whiskerprops = capprops = dict(c='white')
sns.boxplot(data=dataProcessed, orient='horizontal', 
    flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops)

#------PCA

c=7              
pca = PCA(n_components=c,svd_solver='full')
dataAfterPCA = pca.fit_transform(dataFeatures)
print('Explained variance=',pca.explained_variance_ratio_.sum())

#-------- Create linear regression objects  
 
#----Linear regression all features
lregr = LinearRegression()
#----Linear regression PCA features
lregrPCA = LinearRegression()
#----Polynomial regression all features
regrPoly = LinearRegression()
poly = PolynomialFeatures(degree=2)
housePoly=poly.fit_transform(dataFeatures)

n = 0.5

house_train, house_cv, y_train, y_cv = train_test_split(dataFeatures,target,test_size=n, random_state=42)
house_trainPCA, house_cvPCA, y_trainPCA, y_cvPCA = train_test_split(dataAfterPCA,target,test_size=n, random_state=42)
house_trainP, house_cvP, y_trainP, y_cvP = train_test_split(housePoly,target,test_size=n, random_state=42)


lregr.fit(house_train,y_train)
lregrPCA.fit(house_trainPCA,y_trainPCA)
regrPoly.fit(house_trainP,y_trainP)

pred = lregr.predict(house_cv)
predPCA = lregrPCA.predict(house_cvPCA)
predP = regrPoly.predict(house_cvP)

#-------Regularization

ridgeReg = Ridge(alpha=3, normalize=True)
ridgeReg.fit(house_train, y_train)

lassoReg = Lasso(alpha=3, normalize=True)
lassoReg.fit(house_train, y_train)


predRidge = ridgeReg.predict(house_cv)
predLasso = lassoReg.predict(house_cv)


#--------MSE

mse = mean_squared_error(y_cv, pred)
msePCA = mean_squared_error(y_cvPCA, predPCA)
mseP = mean_squared_error(y_cvP, predP)
mseRidge = mean_squared_error(y_cv, predRidge)
mseLasso = mean_squared_error(y_cv, predLasso)

#--------RMSE

rmse = mean_squared_error(y_cv, pred, squared=False)
rmsePCA = mean_squared_error(y_cvPCA, predPCA, squared=False)
rmseP = mean_squared_error(y_cvP, predP, squared=False)
rmseRidge = mean_squared_error(y_cv, predRidge, squared=False)
rmseLasso = mean_squared_error(y_cv, predLasso, squared=False)

#------MAE

mse = mean_absolute_error(y_cv, pred)
msePCA = mean_absolute_error(y_cvPCA, predPCA)
mseP = mean_absolute_error(y_cvP, predP)
mseRidge = mean_absolute_error(y_cv, predRidge)
mseLasso = mean_absolute_error(y_cv, predLasso)

#-----R2
R2_Linear = r2_score(y_cv, pred)
R2_LinearPCA = r2_score(y_cv, predPCA)
R2_Poly = r2_score(y_cv, predP)
R2_Ridge = r2_score(y_cv, predRidge)
R2_Lasso = r2_score(y_cv, predLasso)


#-----Coefficient List
CoeffList = [lregr.coef_,lregrPCA.coef_,regrPoly.coef_,ridgeReg.coef_,lassoReg.coef_]

#-----Interceptor List
IntercList = [lregr.intercept_,lregrPCA.intercept_,regrPoly.intercept_,ridgeReg.intercept_,lassoReg.intercept_]

