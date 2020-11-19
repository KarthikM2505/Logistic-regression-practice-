#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


def image_features():
    names = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness',
             'concavity', 'concave points', 'symmetry', 'fractal dimension']

    columns = ["mean_{}".format(v) for v in names]
    columns.extend(["se_{}".format(v) for v in names])
    columns.extend(["worst_{}".format(v) for v in names])
    return columns


def load_wisconsin_breast_cancer_diagnosis(local_file='wdbc.data'):
    columns = ['ID', 'diagnosis']
    columns.extend(image_features())

    data = pd.read_csv(local_file, index_col=0, header=None,
                       names=columns, dtype={'diagnosis': 'category'})
    y = data.diagnosis.cat.rename_categories(['benign', 'malignant'])
    X = data.drop('diagnosis', axis=1).astype(float)
    return X, y


def load_wisconsin_breast_cancer_prognosis(local_file='wpbc.data'):
    columns = ['ID', 'outcome', 'time']
    outcomes = columns[1:]
    extra_vars = ['tumor_size', 'lymph_node_status']
    columns.extend(image_features())
    columns.extend(extra_vars)

    data = pd.read_csv(local_file, index_col=0, header=None,
                       names=columns, dtype={'outcome': 'category'},
                       na_values=['?'])
    y = data.loc[:, outcomes]
    y.outcome.cat.rename_categories(['nonrecurring', 'recurring'], inplace=True)
    X = data.drop(outcomes, axis=1).astype(float)
    return X, y 


# In[9]:


X_diagnosis, y_diagnosis = load_wisconsin_breast_cancer_diagnosis()

print(X_diagnosis.shape)
print(y_diagnosis.value_counts())


# In[10]:


X_prognosis, y_prognosis = load_wisconsin_breast_cancer_prognosis()

print(X_prognosis.shape)
print(y_prognosis.outcome.value_counts())

plt.hist([
    y_prognosis.time[y_prognosis.outcome == 'recurring'],
    y_prognosis.time[y_prognosis.outcome == 'nonrecurring'],
], label=['recurring', 'nonrecurring'])
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.legend()


# In[ ]:




