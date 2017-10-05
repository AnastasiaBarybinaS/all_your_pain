#!/usr/bin/python
# -*- coding: utf-8 -*-

import plotly
#plotly.tools.set_credentials_file(username='ab_test', api_key='2fl6RY7FsjkyzDv5clm8')

import plotly.plotly as py
from plotly.graph_objs import *
import plotly.figure_factory as ff

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


import numpy as np
import pandas as pd
import scipy as scipy
import scipy.stats as stats

import math

from statsmodels.stats.multicomp import (pairwise_tukeyhsd,MultiComparison)
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


#Count number of unique clients
def uniqueCounter(df,cid):
    print df[cid].nunique();

#Forming draft dataframe
cid = "dimension21"

df = pd.read_csv('a_step2.csv')
df.drop(df.columns[[0,3]], axis=1, inplace=True)
df1 = df[df.dimension21.notnull()].groupby('hip')['dimension21'].nunique()
df1 = pd.DataFrame({'vars': df1.values})
print df1 #uniqueCounter(df,cid)
#step1 = uniqueCounter(df,cid)
#print(step1)
#
# df1 = pd.read_csv('a_step2.csv')
# df1.drop(df1.columns[[0,3]], axis=1, inplace=True)
# step2 = uniqueCounter(df1,cid)
# print(step2)
#
# df2 = pd.read_csv('a_step3.csv')
# df2.drop(df2.columns[[0,3]], axis=1, inplace=True)
# step3 = uniqueCounter(df2,cid)
# print(step3)
#
# df3 = pd.read_csv('a_step4.csv')
# df3.drop(df3.columns[[0,3]], axis=1, inplace=True)
# step4 = uniqueCounter(df,cid)
# print(step4)

#Find out the number and names of experiment branches
def branches(df):
    mass = df['hip'].unique()
    num = df['hip'].nunique()
    return {'mass':mass, 'num':num}
brnch = branches(df)

#find out if the distribution is normal
def normality(data):
    norm = scipy.stats.shapiro(data)
    if norm[1] < 0.05:
        return 'Not Normal distribution'
    elif norm[1] >= 0.05:
        return 'Normal distribution'

#find out if all the groups have the same variation
def homoscedasticity(data):
    bartlett = scipy.stats.bartlett(*data)
    if bartlett[1] < 0.05:
        return 'The variation is different throughout the groups'
    elif bartlett[1] > 0.05:
        return 'All experiment brunches have the same variation'

def ANOVA(data):
    F, p = stats.f_oneway(*data)
    if p < 0.05:
        return 'The median level is significanly different F = %f p-value = %f' % (p, F)
    elif p > 0.05:
        return 'Groups have the same median F = %f p-value = %f' % (p,F)

def kruskalWallis(data):
    F, p = stats.mstats.kruskalwallis(*data)
    if p < 0.05:
        return 'The median level is significanly different F = %f p-value = %f' % (p, F)
    elif p > 0.05:
        return 'Groups have the same median F = %f p-value = %f' % (p,F)

def simpleStatistics(data,n, mass):
    md = np.median(data, axis=1)
    avg = np.average(data, axis=1)
    sd = np.std(data, axis=1)
    var = sd/avg *100

    print "Brunches     ", mass
    print "Mean         ", md
    print "Average      ", avg
    print "St.Dev       ", sd
    print "Variation,%  ", var






#Normal Distribution Tests
#def normalCriteria(x):
    #t.test
    #aov

#Not normal Distribution Tests
#def notNormCriteria():
    # kraskel
    #



#   module calls the function which finds out the number of experiment brunches and their names
#          divide the whole dataset into groups by number of parameters
def divider(n, mas):
    global total #final array
    global observes #number of observations


    total = pd.DataFrame()
    #building the dataframe with format
    # A    B
    #10   10

    for x in range(0, n):
        sdf = df.loc[df['hip']==str(mas[x])]
        sdf = sdf[sdf.dimension21.notnull()].groupby('date')['dimension21'].nunique()
        sdf = pd.DataFrame({'date': sdf.index, 'users': sdf.values})
        total[str(mas[x])] = pd.DataFrame(sdf['users'])
        observes = len(sdf.index);

        #shapiro.test - if the distribution is normal & print out the branch and its distribution type
        print "The brunch " , str(mas[x]) , ' has ', normality(sdf['users'])

    #if the variation is the same throug all the groups
    dlist = np.transpose(total).values.tolist()


    #plot(ff.create_distplot([total[c] for c in total.columns], total.columns))

    print homoscedasticity(dlist)
    print ANOVA(dlist)
    print kruskalWallis(dlist)
    print simpleStatistics(dlist,n,mas)


print divider(brnch["num"], brnch["mass"])




#visualization of brunches
def visual(n, mas):
    global total
    nd = pd.DataFrame(columns=['hip','values'])

    #gdf = df[df.dimension21.notnull()].groupby(['hip', 'date'])['dimension21'].nunique()
    #gdf = pd.DataFrame({'date': gdf.index, 'users': gdf.values, 'hip': gdf.index.levels[0]})

    for param in range(0, n):
        for i in range(0, observes):
            nd.loc[i+observes*param, 'hip'] = str(mas[param])
            nd.loc[i+observes*param, 'values']= total.loc[i,str(mas[param])];

    #print nd.loc[:,'hip']

    '''  
    trace0 = Box(
        y=nd.loc[:,'values'],
        x=nd.loc[:,'hip'],
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        name=['kale','gate'],
        marker=dict(
            color=['#3D9970','#453D32']
        ),
        fillcolor=['#3D9970', '#453D32']
    )
    data = [trace0]
    layout = Layout(
        yaxis=dict(
            title='normalized moisture',
            zeroline=False
        ),
        boxmode='group'
    )
    #fig = go.Figure(data=data, layout=layout)
    #py.plot(fig)
    fig = Figure(data=data, layout=layout)
    plot(fig)
    '''




print visual(brnch["num"], brnch["mass"])



#for i in list(total.columns.values):
#names of columns

#write to csv
#total.to_csv('ttt.csv', sep='\t', encoding='utf-8')
