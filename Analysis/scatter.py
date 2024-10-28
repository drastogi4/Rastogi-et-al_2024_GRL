import numpy as np
import netCDF4 as nc
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import pandas as pd
df1 = pd.read_csv('pr_avg_all_2010-2019.txt', delimiter = " ")
df2 = pd.read_csv('pr_p95_all_2010-2019.txt', delimiter = " ")
df3 = pd.read_csv('25mmdays_all_2010-2019.txt',delimiter = " ")
df4 = pd.read_csv('wetdays_all_2010-2019.txt',delimiter = " ")
fig, ax  = plt.subplots(2, 4, figsize=(11,8))
sns.stripplot(data=df1, x="pr_avg", y="Region",hue="Model",ax=ax[0,0])
sns.stripplot(data=df2, x="p95_avg", y="Region",hue="Model",ax=ax[0,1])
sns.stripplot(data=df3, x="25mmdays_avg", y="Region",hue="Model",ax=ax[0,2])
sns.stripplot(data=df4, x="wetdays_avg", y="Region",hue="Model",ax=ax[0,3])
sns.stripplot(data=df1, x="pr_pc", y="Region",hue="Model",ax=ax[1,0])
sns.stripplot(data=df2, x="p95_pc", y="Region",hue="Model",ax=ax[1,1])
sns.stripplot(data=df3, x="25mmdays_pc", y="Region",hue="Model",ax=ax[1,2])
sns.stripplot(data=df4, x="wetdays_pc", y="Region",hue="Model",ax=ax[1,3])
plt.savefig('scatter.pdf')


