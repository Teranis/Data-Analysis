import os

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np 

import datetime

import regex as re

#Organization (use search with the appropriate number of # followed by a space)
##### Configs
#### Sections: functions, main functions
### Subsections: OD, CC
## Single functions


##### For OD measurements
excel_path = r'E:\Timon\notes\OD_measurements_26_05_23\OD_measurements_26_5_23.xlsx'
exp_name = "Hormone_26_5_23"
no_timepoints = 6
no_perculture = 5
no_cultures = 3
total_pos = no_cultures * no_perculture

##### For Coulter Counter measurements
CC_path = r'E:\Timon\notes\2023_05_26_ASY071_ASY073_ASY075'
CC_exp_name = "Hormone_26_5_23"
culture_names = ['ASY071', 'ASY073', 'ASY075']
custom_order = ['0nM', '2.5nM', '5nM', '10nM', '15nM']

#### Functions
### CC
def import_data_CC(path):
    ## imports data from a single file
    with open(path) as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]

    vols = []
    numbers = []
    bindiam = False
    Binheight = False

    for line in lines:

        if line.startswith('[#Bindiam]'):
            bindiam = True
        if bindiam == True:
            if line.startswith('[Binunits]'):
                bindiam = False
            vols.append(line)
        
        if line.startswith('[#Binheight]'):
            Binheight = True
        if Binheight == True:
            if line.startswith('[end]'):
                Binheight = False
            numbers.append(line)

    del vols[-1]
    del numbers[-1]
    del vols[0]
    del numbers[0]
    for i, vol in enumerate(vols):
        vols[i] = float(vol)
    for i, number in enumerate(numbers):
        numbers[i] = int(number)
    #print(vols, numbers)
    return vols, numbers

def import_all_data_CC(path):
    ## imports CC data from all files in a folder
    data = []
    for file in os.listdir(path):
        if file.endswith('Z2'):
            vols, numbers = import_data_CC(os.path.join(path, file))
            data.append([file, vols, numbers])
    #print(data)
    return data

def plot_CC(entry):
    ## plots CC data
    fig, ax = plt.subplots()
    ax.plot(entry[1], entry[2], marker='x', markersize=4)
    name = entry[0].rstrip(".=#Z2")
    ax.set_title(name)
    ax.set_xlabel('Volume (uL)')
    ax.set_ylabel('Number of cells')
    fig.canvas.manager.set_window_title(CC_exp_name + '_' + name)
    save_path = os.path.join(CC_path, CC_exp_name) + '_' + name + '.png'
    plt.savefig(save_path)
    print('Saved plot to ' + save_path)
    return

def edit_label_CC(label, culture_name):
    ## edits CC label
    label = label.rstrip("_") #idk if this is necessary always, maybe I just did some mistakes when saving my og data set
    label = label.replace(culture_name+'-1_', '') ##### Change this according to your naming scheme
    label = label.replace('_', '.')
    return label

def sort_labels_CC(ax):
    ## sorts out CC labels
    if custom_order != []:
        handles, labels = plt.gca().get_legend_handles_labels()
        sort_list = sorted(range(len(labels)), key=lambda k: custom_order.index(labels[k]))
        ax.legend([handles[idx] for idx in sort_list],[labels[idx] for idx in sort_list])
    return ax

def plot_together_CC(data):
    ## plots CC data together
    for entry in data:
        entry[0] = entry[0].rstrip(".=#Z2")
    for culture_name in culture_names:
        fig, ax = plt.subplots()
        for entry in data:
            if re.match('^.*=?('+culture_name+')', entry[0]):
                label = edit_label_CC(entry[0], culture_name)
                ax.plot(entry[1], entry[2], label=label, marker='x', markersize=4)
        
        ax.set_title(culture_name)
        ax.set_xlabel('Volume (uL)')
        ax.set_ylabel('Number of cells')
        ax.legend()
        ax = sort_labels_CC(ax)
        fig.canvas.manager.set_window_title(CC_exp_name + '_' + culture_name)
        save_path = os.path.join(CC_path, CC_exp_name) + '_' + culture_name + '.png'
        plt.savefig(save_path)
        print('Saved plot to ' + save_path)
    return


### OD
def import_data_OD(path):
    data = pd.read_excel(path)
    return data

def cut_data(data):
    data = data.iloc[1:, 2:]
    return data

def norm_data(data):
    ## normalizing data
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
    ## Calculating time diffs
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
    ## getting culture names
    names = []
    for i in range(0, total_pos, no_perculture):
        names.append(data_metadata.iloc[i+1, 0])
    #print(names)
    return names

def metadata_legend(data_metadata):
    ## getting legend
    legend = []
    for i in range(1, total_pos+1):
        legend.append(str(data_metadata.iloc[i, 1]) + 'nM') #Legend Unit here
    #print(legend)
    return legend



#### Main functions
def test():
    data = import_all_data_CC(CC_path)
    for entry in data:
        print(entry[0].rstrip(".=#Z2"))
### OD
def odnormplot():
    ## creates plots for each culture with normalized OD
    data = import_data_OD(excel_path)
    print(data)
    times = metadata_time(data)
    names = metadata_names(data)
    legend = metadata_legend(data)
    data = cut_data(data)
    data = norm_data(data)

    # creates the figs
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
        plt.savefig(os.path.join(os.path.dirname(excel_path), exp_name + '_' + culturename + '.png'))

    plt.show()

def doublingtime():
    ## calculates doubling time for each culture
    data = import_data_OD(excel_path)
    print(data)
    times = metadata_time(data)
    names = metadata_names(data)
    legend = metadata_legend(data)
    data = cut_data(data)

    results = []
    for i, culturename in enumerate(names):
        for j in range(i*no_perculture, (i+1)*no_perculture, 1):
            doubling_times = []
            for k in range(1, no_timepoints):
                if data.iloc[j, k] > data.iloc[j, k-1]:
                    doubling_times.append((np.log(2) * (times[k] - times[k-1]))/ np.log(data.iloc[j, k] / data.iloc[j, k-1]))
            average = sum(doubling_times) / len(doubling_times)
            print(culturename, legend[j], average)
            results.append([culturename, legend[j], average])
    results = pd.DataFrame(results)
    results.rename(columns={results.columns[0]: 'Culture', results.columns[1]: 'Hormone conc.', results.columns[2]: 'Doubling time'}, inplace=True)
    results.to_excel(os.path.join(os.path.dirname(excel_path), exp_name) + '_doublingtime.xlsx', index=False)


### CC
def coultercounter():
    ## creates CC plots for each exp separately
    data = import_all_data_CC(CC_path)
    for entry in data:
        plt = plot_CC(entry)
    plt.show()

def coulterocunter_culm():
    ## creates CC plots for each exp separately cumulatively
    data = import_all_data_CC(CC_path)
    for entry in data:
        for i, _ in enumerate(entry[2][1:], start=1):
            entry[2][i] = entry[2][i] + entry[2][i-1]
        plt = plot_CC(entry)
    plt.show()

def coulterocunter_together():
    ## creates CC plots for cultures together
    data = import_all_data_CC(CC_path)
    plot_together_CC(data)
    plt.show()

def coulterocunter_together_cum():
    ## creates CC plots for cultures together cumulatively
    data = import_all_data_CC(CC_path)
    for entry in data:
        for i, _ in enumerate(entry[2][1:], start=1):
            entry[2][i] = entry[2][i] + entry[2][i-1]
    
    plot_together_CC(data)
    plt.show()