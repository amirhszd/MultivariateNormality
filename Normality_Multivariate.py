# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:49:27 2019

@author: amir
"""

from scipy.spatial.distance import mahalanobis
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2,beta, probplot
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy import stats
#%%
def multivar_normality(x):
    """
    Normality assessment for multivariate data (data with more than one feature).
    X should have n as number of observations in rows and m as number of features in columns.
    """
    xbar = np.mean(x,axis=0)
    c = np.cov(x.T)
    c_inv = np.linalg.inv(c)
    D2 = np.zeros((x.shape[0]))
    for i in range(x.shape[0]):
        D2[i]=np.dot(np.dot((x[i,:]-xbar).T,c_inv),(x[i,:]-xbar))
    
    # Approach 1
    plt.figure(0)
    D2_sort = np.sort(D2)
    chi = np.sort(np.random.chisquare(x.shape[1],10000))
    chi_q = np.quantile(chi,np.linspace(0.01,0.99,x.shape[0]))
    # Calculatin p value
    var_D2_sort = D2_sort.var(ddof=x.shape[1])
    var_chi_q = chi_q.var(ddof=x.shape[1])
    s = np.sqrt((var_D2_sort + var_chi_q)/2)
    t_chi = (D2_sort.mean() - chi_q.mean())/(s*np.sqrt(2/len(D2_sort)))
    df = x.shape[1]
    p_chi = 1 - stats.t.cdf(t_chi,df=df)    
    
    plt.scatter(chi_q,D2_sort,marker='+')
    x_line = np.linspace(chi.min(),chi.max(),1000)
    y_line = x_line
    plt.plot(x_line,y_line,'r')
    plt.ylabel('Sample Quantile',fontsize=14)
    plt.xlabel('Theoretical Quantile',fontsize=14)
    plt.title(r"$Chi$-squared Method, p-value=%0.3f"%(p_chi),fontsize=14)
    plt.legend([r'y=x line',r'Samples']) 

    # Approach 2
    plt.figure(1)
    a=x.shape[1]/2
    b=(x.shape[0]-x.shape[1]-1)/2
    bet = np.sort(np.random.beta(a,b,10000))
    bet = np.quantile(bet,np.linspace(0.01,0.99,x.shape[0]))
    u = np.sort((len(x)*D2)/((len(x)-1)**2))
    # Calculatin p value
    var_u = u.var(ddof=x.shape[1])
    var_bet = bet.var(ddof=x.shape[1])
    s = np.sqrt((var_u + var_bet)/2)
    t_bet = (u.mean() - bet.mean())/(s*np.sqrt(2/len(u)))
    df = x.shape[1]
    p_bet = 1 - stats.t.cdf(t_bet,df=df)    

    plt.scatter(bet,u,marker='+')
    x_line = np.linspace(bet.min(),bet.max(),1000)
    y_line = x_line
    plt.plot(x_line,y_line,'r')
    plt.ylabel('Sample Quantile',fontsize=14)
    plt.xlabel('Theoretical Quantile',fontsize=14)
    plt.title('Small\'s Method, p-value=%0.3f'%(p_bet),fontsize=14)
    plt.legend([r'y=x line',r'Samples'])

    return D2_sort
