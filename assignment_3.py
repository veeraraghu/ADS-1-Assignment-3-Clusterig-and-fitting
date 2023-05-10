# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:21:06 2023

@author: Acer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import cluster_tools as ct
import errors as err
import importlib


def read_data(file_path):
    data=pd.read_csv(file_path,skiprows=4)
    data=data.set_index('Country Name',drop=True)
    data=data.loc[:,'1960':'2021']
    return data
def transpose(data):
    data_tr=data.transpose()
    
    return data_tr

def corr_scattermatrix(data):
    corr = data.corr()
    print(corr) 
    plt.figure(figsize=(10, 10))
    # fig, ax = plt.subplots()
    plt.matshow(corr, cmap='coolwarm')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()
    plt.show()

    pd.plotting.scatter_matrix(data, figsize=(12, 12), s=5, alpha=0.8)
    plt.show()
    
    return

def n_clusters(data,data_norm,a,b):
    
    n_clusters=[]
    cluster_score=[]
    
    
    for ncluster in range(2, 10):
        
        # set up the clusterer with the number of expected clusters
        kmeans = cluster.KMeans(n_clusters=ncluster)

        # Fit the data, results are stored in the kmeans object
        kmeans.fit(data_norm)     # fit done on x,y pairs

        labels = kmeans.labels_
        
        # extract the estimated cluster centres
        cen = kmeans.cluster_centers_

        # calculate the silhoutte score 
        print(ncluster, skmet.silhouette_score(data, labels))
        
        n_clusters.append(ncluster)
        cluster_score.append(skmet.silhouette_score(data, labels))
        
    n_clusters=np.array(n_clusters)
    cluster_score=np.array(cluster_score)
        
    best_ncluster=n_clusters[cluster_score==np.max(cluster_score)]
    print('best n clusters',best_ncluster[0])
    
    return best_ncluster[0]
        
        
    
def clusters_and_centers(df_norm,ncluster,a,b):
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_norm)     # fit done on x,y pairs

    labels = kmeans.labels_
    df_norm['labels']=labels
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_

    cen = np.array(cen)
    xcen = cen[:, 0]
    ycen = cen[:, 1]


    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))

    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_norm[a], df_norm[b], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel(f"GDP({a})")
    plt.ylabel(f"GDP({b})")
    plt.show()

    print(cen)

    
    return 
    

df_co2 = read_data("gdp_per_capita.csv")
print(df_co2.describe())

df_co2_tr=transpose(df_co2)
print(df_co2_tr.head())

df_co3 = df_co2[["1970", "1980", "1990", "2000", "2010",'2020']]
print(df_co3.describe())

corr_scattermatrix(df_co3)
a="1990"
b="2020"
df_ex = df_co3[[a, b]]  # extract the two columns for clustering


df_ex = df_ex.dropna()  # entries with one nan are useless
print(df_ex.head())

# normalise, store minimum and maximum
df_norm, df_min, df_max = ct.scaler(df_ex)

print()
print("n  score")
# loop over number of clusters
ncluster=n_clusters(df_ex,df_norm,a,b)

clusters_and_centers(df_norm, ncluster,a,b)

clusters_and_centers(df_ex, ncluster,a,b)

print(df_ex[df_ex['labels']==ncluster-1])


df_co2_tr=df_co2_tr.loc[:,'Brazil']
df_co2_tr=df_co2_tr.dropna(axis=0) 
print('Transpose')
print(df_co2_tr.head())


df_gdp=pd.DataFrame()

df_gdp['Year']=pd.DataFrame(df_co2_tr.index)
df_gdp['GDP']=pd.DataFrame(df_co2_tr.values)

print(df_gdp.head())

df_gdp.plot("Year", "GDP")
plt.show()


df_gdp["Year"] = pd.to_numeric(df_gdp["Year"])

# param, covar = opt.curve_fit(exponential, df_gdp["Year"], df_gdp["GDP"], p0=(1.2e12, 0.03))


def logistic(t, n0, g, t0): 
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    
    f = n0 / (1 + np.exp(-g*(t - t0)))
    
    return f

importlib.reload(opt)

param, covar = opt.curve_fit(logistic, df_gdp["Year"], df_gdp["GDP"], 
                             p0=(1.2e12, 0.03, 1990.0))

sigma = np.sqrt(np.diag(covar))

df_gdp["fit"] = logistic(df_gdp["Year"], *param)

df_gdp.plot("Year", ["GDP", "fit"])
plt.show()

print("turning point", param[2], "+/-", sigma[2])
print("GDP at turning point", param[0]/1e9, "+/-", sigma[0]/1e9)
print("growth rate", param[1], "+/-", sigma[1])



year = np.arange(1960, 2031)
forecast = logistic(year, *param)

plt.figure()
plt.plot(df_gdp["Year"], df_gdp["GDP"], label="GDP")
plt.plot(year, forecast, label="forecast")

plt.xlabel("year")
plt.ylabel("GDP")
plt.legend()
plt.show()

import errors as err

low, up = err.err_ranges(year, logistic, param, sigma)

plt.figure()
plt.plot(df_gdp["Year"], df_gdp["GDP"], label="GDP")
plt.plot(year, forecast, label="forecast")

plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("GDP")
plt.legend()
plt.show()

print(logistic(2030, *param)/1e9)
print(err.err_ranges(2030, logistic, param, sigma))

# assuming symmetrie estimate sigma
gdp2030 = logistic(2030, *param)/1e9

low, up = err.err_ranges(2030, logistic, param, sigma)
sig = np.abs(up-low)/(2.0 * 1e9)
print()
print("GDP 2030", gdp2030, "+/-", sig)









