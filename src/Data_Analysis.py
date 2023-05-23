import os

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np 

import datetime

excel_path = r'H:\Timon\notes\OD_measurement_16_5_23\OD_measurements_16_5_23.xlsx'
exp_name = "Hormone_16_5_23"
no_timepoints = 6
no_perculture = 4
no_cultures = 6
total_pos = no_cultures * no_perculture

def import_data(path):
    data = pd.read_excel(path)
    return data

def cut_data(data):
    data = data.iloc[1:, 2:]
    return data

def norm_data(data):
    ##normalizing data
    norm = []
    for i in range(total_pos):
        norm.append(1 / data.iloc[i, 0])
    #print(norm)
    for i in range(total_pos):
        for j in range(no_timepoints):
            data.iloc[i, j] = data.iloc[i, j] * norm[i]
    print(data)
    return data

def metadata_time(data_metadata):
    ###Calculating time diffs
    times = [0]
    for i in range(2, no_timepoints + 1):
        delta = 0
        date = datetime.date.today()
        datetime1 = datetime.datetime.combine(date, data_metadata.iloc[0, i+1])
        datetime2 = datetime.datetime.combine(date, data_metadata.iloc[0, i])
        delta = datetime1 - datetime2
        times.append(times[-1] + delta.total_seconds() / 3600)
    #print(times)
    return times

def metadata_names(data_metadata):
    ###getting culture names
    names = []
    for i in range(0, total_pos, no_perculture):
        names.append(data_metadata.iloc[i+1, 0])
    #print(names)
    return names

def metadata_legend(data_metadata):
    ###getting legend
    legend = []
    for i in range(1, total_pos+1):
        legend.append(str(data_metadata.iloc[i, 1]) + 'nM') #Legend Unit here
    #print(legend)
    return legend

def run():

    data = import_data(excel_path)
    print(data)
    times = metadata_time(data)
    names = metadata_names(data)
    legend = metadata_legend(data)
    data = cut_data(data)
    data = norm_data(data)

    ##creates the figs
    for i, culturename in enumerate(names):
        fig, ax = plt.subplots()
        for j in range(i*no_perculture, (i+1)*no_perculture, 1):
            ax.plot(times, data.iloc[j], marker='o', label=legend[j])
        ax.set_title(culturename)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Normalized Optical Density')
        ax.set_yscale('log')
        ax.legend()
        fig.canvas.manager.set_window_title(exp_name + '_' + culturename)
        #print(os.path.join(os.path.dirname(excel_path), exp_name + '_' + culturename))
        plt.savefig(os.path.join(os.path.dirname(excel_path), exp_name + '_' + culturename + '.png'), format = 'png', )
        #return plt

    plt.show()