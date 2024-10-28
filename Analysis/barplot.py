import numpy as np
import netCDF4 as nc
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import pandas as pd

def plot(df,region):
    dfreg = df[(df.reg==region)]
    order = ["SRCNN-MSE","SRCNN-MSE-EL","SRCNN-MSE-EL-Proc","SRCNN-EXP-EL-Proc","SRCNN-QT-EL-Proc"]
    dfreg.loc[:,'Model'] = pd.Categorical(dfreg['Model'], categories=order, ordered=True)
    sorted_df = dfreg.sort_values(by='Model')
    g = sns.catplot(
        data=sorted_df, kind="bar",
        y="Model", x="wetdays_avg", hue="Years",
        palette="rocket_r")
    plt.savefig(f'{var}_{region}_bar.pdf')

def main():
    df = pd.read_csv(f'./txtfiles/{var}_2010-2019.txt', delimiter = " ")
    regions = ("CONUS","midwest","northcentral","nothreast","northwest","southcentral","southeast","southwest")
    for region in regions:
        plot(df,region)

if __name__ == "__main__":
    main()
