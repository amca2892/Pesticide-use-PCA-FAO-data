# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:08:23 2022

@author: amca2
"""

#%% import libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sb

#%% get right directory
os.chdir(r"D:\Data_Mining\Data_Mining_Ale")

#%% create dataframe
df_h = pd.read_csv("FAOSTAT_data_4-12-2022.csv")
df_y = pd.read_csv("yield.csv")
df_fu_ins = pd.read_csv('fung+ins.csv')
df_fu_ins = df_fu_ins.pivot(index=["Año","Área"], columns='Producto', values="Valor")
df_fu_ins = df_fu_ins.reset_index()
df_gmo = pd.read_csv('gmo.csv', sep=';')
df_y = df_y[df_y['Elemento'] == 'Rendimiento']
df_h['Key'] = df_h.Año.map(str) + df_h.Área
df_y['Key'] = df_y.Año.map(str) + df_y.Área
df_fu_ins['Key'] = df_fu_ins.Año.map(str) + df_fu_ins.Área
df_fu_ins = df_fu_ins.rename(columns={'Fungicidas y bactericidas':'Fungicides', 'Insecticidas': 'Insecticides' })



#%% group yields by country_year

df_y_g = df_y.groupby(["Key"]).agg({"Valor": ['mean']})
df_y_g.columns = ['yield_mean']
df_y_g = df_y_g.rename_axis('Key').reset_index()

df_h_g = df_h.groupby(["Key"]).agg({"Valor": ['mean']})
df_h_g.columns = ['Herbicides']
df_h_g = df_h_g.rename_axis('Key').reset_index()



#%% join dataframe and assign GMO label
import numpy as np

df = df_y_g.join(df_h_g.set_index('Key'), on='Key')
df = df.join(df_fu_ins.set_index('Key'), on='Key')

df['Country'] = df['Key'].str.slice(4)
df['Year'] = df['Key'].str.slice(0,4)
df = df.join(df_gmo.set_index('Country'), on='Country')
#df = df.drop(['Country'], axis=1)


#%% correlation matrix

df_corr = df[['yield_mean', 'Herbicides', 'Insecticides', 'Fungicides']]

corr = df_corr.corr()
sb.heatmap(corr, cmap="Blues", annot=True)
#%% create PCA

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#df to numpy
X = df[['yield_mean', 'Herbicides', 'Fungicides', 'Insecticides']].to_numpy()
#scale X
X = StandardScaler().fit_transform(X)
pca = PCA()
pca.fit(X)
print(pca.explained_variance_ratio_)
#pca to df 
x_pca = pca.fit_transform(X)
pca_df = pd.DataFrame(data=x_pca, columns=['pc1', 'pc2', 'pc3', 'pc4'])

#%% final dataframe

df = pd.concat((df, pca_df), axis=1)

#%% plot PCA


sb.relplot(data=df, x='pc1', y='pc2', hue='Country', style='GMO', aspect=1.61)
plt.show()

#%% create Dendrogram

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



X = pca_df.to_numpy()

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
#%% general plots

df_yield_by_country = df.pivot(index="Year", columns='Country', values="yield_mean")
#df_yield_by_country = df_yield_by_country.reset_index(level=0)
df_h_by_country = df.pivot(index="Year", columns='Country', values="Herbicides")
df_f_by_country = df.pivot(index="Year", columns='Country', values="Fungicides")
df_i_by_country = df.pivot(index="Year", columns='Country', values="Insecticides")

plt.xticks(rotation=90)
sb.lineplot(data=df_yield_by_country, dashes=False)
plt.show()

plt.xticks(rotation=90)
sb.lineplot(data=df_i_by_country, dashes=False)
plt.show()

plt.xticks(rotation=90)
sb.lineplot(data=df_f_by_country, dashes=False)
plt.show()

plt.xticks(rotation=90)
sb.lineplot(data=df_h_by_country, dashes=False)
plt.show()
