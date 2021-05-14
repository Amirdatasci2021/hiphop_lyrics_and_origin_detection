#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import random
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text


# In[2]:


#read in the csv, yes the df is right their but I want to make sure nothing funky happens 
pre_df=pd.read_csv('./assets/NLP_full.csv')
#toss it around
pre_df = shuffle(pre_df)
rappers=pre_df['artist'].unique()
rappers=list(rappers.flatten())
#pushes it to numpy array and then a flat list with tuples


# In[3]:


#select the random n targets 
n_rappers=4
n_exp=1
#this will execute the combinations we want just one at time which yes is counter-intutive since later will be running
#the script 300 times to create unique test sets
ps_comb = combinations(rappers, n_rappers) 
ps_targets=[i for i in list(ps_comb)]
ps_targets_indf=(random.sample(ps_targets, n_exp))
#now stick it in a dataframe
ps_df = pd.DataFrame(ps_targets_indf, columns =['first','second','third','fourth'])
spacer=", "
ps_df['artist_selection']=(ps_df['first']+spacer+ps_df['second']+spacer+ps_df['third']+spacer+ps_df['fourth'])


# In[4]:


one=ps_df['first'].unique()
one=list(one.flatten())
two=ps_df['second'].unique()
two=list(two.flatten())
three=ps_df['third'].unique()
three=list(three.flatten())
four=ps_df['fourth'].unique()
four=list(four.flatten())


# In[5]:


checks=(set(one+two+three+four))
checks=list(checks)


# In[6]:


#This script gives me the difference between my two lists and provides simple list of artists to drop
#https://www.geeksforgeeks.org/python-difference-two-lists/

def diff(li1, li2):
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))
checks_out=diff(checks,rappers)


# In[7]:


sel_df=pd.read_csv('./assets/NLP_full.csv')
sel_columns=sel_df.columns


# In[8]:


sel_df.drop(columns=['length','expl_bool', 'total_expl',
       'grp_1', 'grp_2', 'tb_polarity', 'tb_subjectivity', 'vader_neg',
       'vader_neu', 'vader_pos', 'compound'],inplace=True)


# In[9]:


for each in checks_out:
    sel_df.drop(sel_df[sel_df.artist==each].index, inplace=True)


# In[10]:


trip_check=sel_df.artist.unique()
trip_check=list(trip_check.flatten())
#when you triple check something it means you really don't want to screw up lol


# In[11]:


name_file=str(sel_df.artist.unique().flatten())


# In[12]:


name_file=''.join(each for each in name_file if each.isalnum())


# In[13]:


name_file=name_file +'.csv'
name_file="./data/"+name_file


# In[14]:


sel_df.drop(columns='number_words',inplace=True)


# In[15]:


sel_df.to_csv(name_file)
#I always forget to get rid of the index 


# In[ ]:





# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




