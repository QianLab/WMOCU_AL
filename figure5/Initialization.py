
import numpy as np
from scipy.special import expit
import pandas as pd
from numpy.random import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#%%

def pz_theta_model(x, theta_r):
    # x can be np.array of size 2 or a list of two np.array x1 and x2
    xnum = x.shape[0]
    pzx_theta = np.ones((2, xnum))
    pzx1_theta = theta_r[x]
    pzx0_theta = 1 - pzx1_theta
    pzx_theta[0, :] = pzx0_theta
    pzx_theta[1, :] = pzx1_theta
    return pzx_theta

py_eq_z = None

def Initialization(xnum, thetanum):
    df = pd.read_csv('knowledge.csv')
    #df = df[df['UNS'].isin(['High', 'Very Low', 'very_low'])]
    df.loc[ df['UNS'].isin(['High']), 'UNS'] = 1
    df.loc[ df['UNS'].isin(['Middle']), 'UNS'] = 1
    df.loc[ df['UNS'].isin(['Low']), 'UNS'] = 0
    df.loc[ df['UNS'].isin(['Very Low']), 'UNS'] = 0
    df.loc[ df['UNS'].isin(['very_low']), 'UNS'] = 0
    #print(df.groupby('UNS').count())
#    collist = ['STG', 	'SCG',	'STR',	'LPR',	'PEG','UNS']
    #df.hist(column = collist[0], by = collist[5], sharex = True, sharey = True, bins = 4)
    #df.hist(column = collist[1], by = collist[5], sharex = True, sharey = True, bins = 4)
    #df.hist(column = collist[2], by = collist[5], sharex = True, sharey = True, bins = 4)
    #df.hist(column = collist[3], by = collist[5], sharex = True, sharey = True, bins = 4)
    #df.hist(column = collist[4], by = collist[5], sharex = True, sharey = True, bins = 4)
    
    df2 = df.iloc[:, [0, 4, 5]]
    df3 = df2[df2['UNS'].isin([0])]
    df4 = df2[df2['UNS'].isin([1])]
    X_train1, X_test1, y_train1, y_test1 = train_test_split(df3.iloc[:, :-1], df3.iloc[:, -1], test_size = 150)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(df4.iloc[:, :-1], df4.iloc[:, -1], test_size = 150)
    X_train = pd.concat([X_train1, X_train2])
    X_test = pd.concat([X_test1, X_test2])
    y_train = pd.concat([y_train1, y_train2])
    y_test = pd.concat([y_test1, y_test2])
    #    X_pool, X_test, y_pool, y_test = train_test_split(X_test, y_test, train_size = 1000)
    X_pool = X_test
    y_pool = y_test
    y_pool = y_pool.values
    bins = 4
    
    xtik1 = np.linspace(0, 1, bins+1)
    xtik2 = np.linspace(0, 1, bins+1)
    
    b1 = np.digitize(X_pool.iloc[:, 0], xtik1)
    b2 = np.digitize(X_pool.iloc[:, 1], xtik2)
    
    bidx = (b1-1)*bins+b2-1
    xspace = bidx
    yspace = y_pool
    
#    error_count = 0
    count0 = np.zeros(bins**2)
    count1 = np.zeros(bins**2)
    
    for i, x in enumerate(xspace):
        if yspace[i] == 0:
            count0[x] +=1
        else:
            count1[x] += 1
            
    OBC = (count0<count1).astype(int)
    
#    error_list = np.minimum(count0, count1)
    
#    bayesian_error = np.sum(error_list)/len(xspace)
    hyperlist = np.ones((2, bins**2))
    alist = np.ones(bins**2)
    blist = np.ones(bins**2)
    bad = False
    if bad:
        OBC = 1-OBC
    
    for i, label in enumerate(OBC):
        if label == 0:
            alist[i] = 2
            blist[i] = 5
        else:
            alist[i] = 5
            blist[i] = 2
    alist[0:8] = 10
    blist[0:8] = 10
    
#    blist[5:10] = 2
    hyperlist[1, :] = alist
    hyperlist[0, :] = blist
    
    return xspace, yspace, hyperlist

classnum = 2
multiclass = True


#%%
df = pd.read_csv('knowledge.csv')
#df = df[df['UNS'].isin(['High', 'Very Low', 'very_low'])]
df.loc[ df['UNS'].isin(['High']), 'UNS'] = 1
df.loc[ df['UNS'].isin(['Middle']), 'UNS'] = 1
df.loc[ df['UNS'].isin(['Low']), 'UNS'] = 0
df.loc[ df['UNS'].isin(['Very Low']), 'UNS'] = 0
df.loc[ df['UNS'].isin(['very_low']), 'UNS'] = 0
#print(df.groupby('UNS').count())
collist = ['STG', 	'SCG',	'STR',	'LPR',	'PEG','UNS']
#df.hist(column = collist[0], by = collist[5], sharex = True, sharey = True, bins = 4)
#df.hist(column = collist[1], by = collist[5], sharex = True, sharey = True, bins = 4)
#df.hist(column = collist[2], by = collist[5], sharex = True, sharey = True, bins = 4)
#df.hist(column = collist[3], by = collist[5], sharex = True, sharey = True, bins = 4)
#df.hist(column = collist[4], by = collist[5], sharex = True, sharey = True, bins = 4)

df2 = df.iloc[:, [0, 4, 5]]
df3 = df2[df2['UNS'].isin([0])]
df4 = df2[df2['UNS'].isin([1])]
X_train1, X_test1, y_train1, y_test1 = train_test_split(df3.iloc[:, :-1], df3.iloc[:, -1], test_size = 150)
X_train2, X_test2, y_train2, y_test2 = train_test_split(df4.iloc[:, :-1], df4.iloc[:, -1], test_size = 150)
X_train = pd.concat([X_train1, X_train2])
X_test = pd.concat([X_test1, X_test2])
y_train = pd.concat([y_train1, y_train2])
y_test = pd.concat([y_test1, y_test2])
#    X_pool, X_test, y_pool, y_test = train_test_split(X_test, y_test, train_size = 1000)
X_pool = X_train
y_pool = y_train
bins = 4

xtik1 = np.linspace(0, 1, bins+1)
xtik2 = np.linspace(0, 1, bins+1)

b1 = np.digitize(X_pool.iloc[:, 0], xtik1)
b2 = np.digitize(X_pool.iloc[:, 1], xtik2)

bidx = (b1-1)*bins+b2-1
xspace = bidx
yspace = y_pool

error_count = 0
count0 = np.zeros(bins**2)
count1 = np.zeros(bins**2)

for i, x in enumerate(xspace):
    if yspace.iloc[i] == 0:
        count0[x] +=1
    else:
        count1[x] += 1
        
error_list = np.minimum(count0, count1)

bayesian_error = np.sum(error_list)/len(xspace)

alphalist = np.ones(bins**2)
        
            


##%%
#df = pd.read_csv('dataset.csv')
#clist = df.columns
#df.hist(column =clist[2], by = clist[0])