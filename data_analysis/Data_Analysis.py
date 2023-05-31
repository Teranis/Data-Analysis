from math import e, log
import os

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np 

import datetime

import regex as re

import json

import math

from scipy.optimize import curve_fit

import matplotlib.cm as mcolors

import subprocess

#Organization (use search with the appropriate number of # followed by a space)
##### Configs (Outsourced to config.json))
#### Sections: functions, main functions
### Subsections: OD, CC
## Single functions






#### Functions
### ConfigLoad
def importconfigCC():
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_file_path = os.path.join(parent_dir, 'configCC.json')
    with open(config_file_path, 'r') as input_file:
        config_raw = input_file.read()
    config_raw = config_raw.replace('\\','/')
    config = json.loads(config_raw)
    CC_path = config['CC_path']
    CC_exp_name = config['CC_exp_name']
    culture_names = config['CC_culture_names']
    custom_order = config['CC_custom_order']
    CC_norm_data = config['CC_norm_data']
    CC_culm = config['CC_culm']
    return CC_path, CC_exp_name, culture_names, custom_order, CC_norm_data, CC_culm

def importconfigOD():
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_file_path = os.path.join(parent_dir, 'configOD.json')
    with open(config_file_path, 'r') as input_file:
        config_raw = input_file.read()
    config_raw = config_raw.replace('\\','/')
    config = json.loads(config_raw)
    excel_folder_path = config['OD_excel_path']
    exp_name = config['OD_exp_name']
    no_timepoints = config['OD_no_timepoints']
    no_perculture = config['OD_no_perculture']
    no_cultures = config['OD_no_cultures']
    total_pos = no_cultures * no_perculture
    OD_norm_data = config['OD_norm_data']
    use_fit = config['OD_use_fit']
    OD_exp_fit = config['OD_exp_fit']
    for file_name in os.listdir(excel_folder_path):
        if re.search(r"(=?(measure))(=?(.*\.xlsx)$)", file_name):
            excel_path = os.path.join(excel_folder_path, file_name)
    print(excel_path)
    return excel_path, exp_name, no_timepoints, no_perculture, no_cultures, total_pos, OD_norm_data, use_fit, OD_exp_fit

def openpath():
    os.chdir("..")

def openexpath():
    path = os.path.abspath("..")
    subprocess.Popen(['explorer', path])

def openconfig():
    for filename in ['config.CC', 'config.OD']:
        try:
            subprocess.run(['start', '', filename], shell=True)  # For Windows
        except FileNotFoundError:
            try:
                subprocess.run(['xdg-open', filename])  # For Linux
            except FileNotFoundError:
                try:
                    subprocess.run(['open', filename])  # For macOS
                except FileNotFoundError:
                    print("Unable to open the file.")

def openexamplexlsx():
    for filename in ['example.xlsx']:
        try:
            subprocess.run(['start', '', filename], shell=True)  # For Windows
        except FileNotFoundError:
            try:
                subprocess.run(['open', '-a', 'Microsoft Excel', filename])  # For macOS
            except FileNotFoundError:
                print("Unable to open the file with Excel.")
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
    for i, vol in enumerate(vols):
        vols[i] = (math.sqrt(vol)**3)*4/3
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

def norm_data_cc(data, CC_norm_data):
    if CC_norm_data:
        for entry in data:
            total = sum(entry[2])
            for i, number in enumerate(entry[2]):
                entry[2][i] = number / total
    return data
                
def edit_label_CC(label, culture_name):
    ## edits CC label
    label = label.rstrip("_") #idk if this is necessary always, maybe I just did some mistakes when saving my og data set
    label = label.replace(culture_name+'-1_', '') ##### Change this according to your naming scheme
    label = label.replace('_', '.')
    return label

def sort_labels_CC(ax, custom_order):
    ## sorts out CC labels
    if custom_order != []:
        handles, labels = plt.gca().get_legend_handles_labels()
        sort_list = sorted(range(len(labels)), key=lambda k: custom_order.index(labels[k]))
        ax.legend([handles[idx] for idx in sort_list],[labels[idx] for idx in sort_list])
    return ax

def plot_CC(entry, CC_exp_name, CC_path, CC_norm_data):
    ## plots CC data
    fig, ax = plt.subplots()
    ax.plot(entry[1], entry[2], marker='x', markersize=4)
    name = entry[0].rstrip(".=#Z2")
    ax.set_title(name)
    ax.set_xlabel('Volume (uL)')
    if CC_norm_data == True:
        ax.set_ylabel('Fraction of cells')
    else:
        ax.set_ylabel('Number of cells')
    ax.grid(True)
    fig.canvas.manager.set_window_title(CC_exp_name + '_' + name)
    save_path = os.path.join(CC_path, CC_exp_name) + '_' + name + '.png'
    plt.savefig(save_path)
    print('Saved plot to ' + save_path)
    return 

def plot_together_CC(data, culture_names, CC_norm_data, custom_order, CC_exp_name, CC_path):
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
        if CC_norm_data == True:
            ax.set_ylabel('Fraction of cells')
        else:
            ax.set_ylabel('Number of cells')
        ax.legend()
        ax = sort_labels_CC(ax, custom_order)
        ax.grid(True)
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

def norm_data(data, total_pos, no_timepoints):
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

def metadata_time(data_metadata, no_timepoints):
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

def metadata_names(data_metadata, total_pos, no_perculture):
    ## getting culture names
    names = []
    for i in range(0, total_pos, no_perculture):
        names.append(data_metadata.iloc[i+1, 0])
    #print(names)
    return names

def metadata_legend(data_metadata, total_pos):
    ## getting legend
    legend = []
    for i in range(1, total_pos+1):
        legend.append(str(data_metadata.iloc[i, 1]) + 'nM') #Legend Unit here
    #print(legend)
    return legend

def fit_curve(time, start_OD, D):
    ## exp fit
    OD = start_OD*np.exp(time*D)
    return OD

def fit_curve_lin(time, start_OD, D):
    ## linear fit
    OD = start_OD + time*D
    return OD

def fitting(fit_curve, OD, time, start_OD, startval):
## fitting the data
    D, cov_ma = curve_fit(lambda time, D: fit_curve(time, start_OD, D), time, OD, startval)
    if OD_exp_fit == True:
        D = log(2)/D
    return D

#### Main functions
#def test():
#    data = import_all_data_CC(CC_path)
#    for entry in data:
#        print(entry[0].rstrip(".=#Z2"))
### OD
def odplot():
    ## creates plots for each culture with normalized OD
    excel_path, exp_name, no_timepoints, no_perculture, no_cultures, total_pos, OD_norm_data, use_fit, OD_exp_fit = importconfigOD()
    data = import_data_OD(excel_path)
    print(data)
    times = metadata_time(data, no_timepoints)
    names = metadata_names(data, total_pos, no_perculture)
    legend = metadata_legend(data, total_pos)
    data = cut_data(data)
    if OD_norm_data == True:
        data = norm_data(data, total_pos, no_timepoints)

    # creates the figs
    for i, culturename in enumerate(names):
        fig, ax = plt.subplots()
        for j in range(i*no_perculture, (i+1)*no_perculture, 1):
            ax.plot(times, data.iloc[j], marker='o', label=legend[j])
        ax.set_title(culturename)
        ax.set_xlabel('Time (h)')
        if OD_norm_data == True:
            ax.set_ylabel('Normalized Optical Density')
        else:
            ax.set_ylabel('Optical Density')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
        fig.canvas.manager.set_window_title(exp_name + '_' + culturename)
        #print(os.path.join(os.path.dirname(excel_path), exp_name + '_' + culturename))
        plt.savefig(os.path.join(os.path.dirname(excel_path), exp_name + '_' + culturename + '.png'))

    plt.show()

def doublingtime():
    ## calculates doubling time for each culture
    excel_path, exp_name, no_timepoints, no_perculture, no_cultures, total_pos, OD_norm_data, use_fit, OD_exp_fit = importconfigOD()
    data = import_data_OD(excel_path)
    print(data)
    times = metadata_time(data, no_timepoints)
    names = metadata_names(data, total_pos, no_perculture)
    legend = metadata_legend(data, total_pos)
    data = cut_data(data)
    print(data) 
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
    if use_fit == True:
        results_avg = results.copy()
        results = []
        if OD_norm_data == True:
            data = norm_data(data, total_pos, no_timepoints)
        for i, culturename in enumerate(names):
            for j in range(i*no_perculture, (i+1)*no_perculture, 1):
                ODs = []
                for k in range(no_timepoints):
                    ODs.append(data.iloc[j, k])
                if OD_exp_fit == True:
                    D = fitting(fit_curve, ODs, np.array(times), data.iloc[j, 1], results_avg.iloc[j][2])
                else:
                    D = fitting(fit_curve_lin, ODs, np.array(times), data.iloc[j, 1], results_avg.iloc[j][2])
                results.append([culturename, legend[j], D])
        results = pd.DataFrame(results)
        results.rename(columns={results.columns[0]: 'Culture', results.columns[1]: 'Hormone conc.', results.columns[2]: 'Doubling time'}, inplace=True)
        results.to_excel(os.path.join(os.path.dirname(excel_path), exp_name) + '_doublingtime_fit.xlsx', index=False)
    color_map = mcolors.get_cmap('tab10')
    colors = [color_map(i) for i in range(no_perculture)]
    print(len(colors))
    for i, culturename in enumerate(names):
        fig, ax = plt.subplots()
        for j in range(i*no_perculture, (i+1)*no_perculture, 1):
            scatter = ax.scatter(times, data.iloc[j], marker='x', label=results.iloc[j][1], color=colors[j-i*no_perculture])
            y = []
            for k, _ in enumerate(times):
                if OD_exp_fit == True:
                    y.append(fit_curve(times[k], data.iloc[j, 0], log(2)/results.iloc[j][2]))
                else:
                    y.append(fit_curve_lin(times[k], data.iloc[j, 0], results.iloc[j][2]))
            ax.plot(times, y, color=colors[j-i*no_perculture])
        ax.set_title(culturename)
        ax.set_xlabel('Time (h)')
        if OD_norm_data == True:
            ax.set_ylabel('Normalized Optical Density')
        else:
            ax.set_ylabel('Optical Density')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
        fig.canvas.manager.set_window_title(exp_name + '_' + culturename + '_fit')
        #print(os.path.join(os.path.dirname(excel_path), exp_name + '_' + culturename))
        plt.savefig(os.path.join(os.path.dirname(excel_path), exp_name + '_' + culturename + '_fit.png'))
    plt.show()



### CC

def coulterocunter():
    ## creates CC plots for each exp separately cumulatively
    CC_path, CC_exp_name, culture_names, custom_order, CC_norm_data, CC_culm = importconfigCC()
    data = import_all_data_CC(CC_path)
    data = norm_data_cc(data, CC_norm_data)
    for entry in data:
        if CC_culm == True:
            for i, _ in enumerate(entry[2][1:], start=1):
                entry[2][i] = entry[2][i] + entry[2][i-1]
        plot_CC(entry, CC_exp_name, CC_path, CC_norm_data)
    plt.show()

def coulterocunter_together():
    ## creates CC plots for cultures together cumulatively
    CC_path, CC_exp_name, culture_names, custom_order, CC_norm_data, CC_culm = importconfigCC()
    data = import_all_data_CC(CC_path)
    data = norm_data_cc(data, CC_norm_data)
    if CC_culm == True:
        for entry in data:
            for i, _ in enumerate(entry[2][1:], start=1):
                entry[2][i] = entry[2][i] + entry[2][i-1]
    plot_together_CC(data, culture_names, CC_norm_data, custom_order, CC_exp_name, CC_path)
    plt.show()