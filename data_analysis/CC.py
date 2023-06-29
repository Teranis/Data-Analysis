from calendar import c
import os
from re import A
import matplotlib.pyplot as plt
import regex as re
import math
from scipy import stats
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import numpy as np
import pandas as pd
import operator
#from data_analysis.core import saveexcel, labelreorg, getcolormap
#from data_analysis.configload import importconfigCC
#from data_analysis.core import sort_labels as sort_labels_CC
from core import sort_labels as sort_labels_CC
from core import labelreorg, getcolormap, saveexcel, sorting_dataframe, printl
from configload import importconfigCC
import copy
import matplotlib.cbook

#from data_analysis.core import sorting_dataframe
#import seaborn as sns
### small
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
        vols[i] = (4/3)*math.pi*(vol)**(3)
    return vols, numbers

def import_all_data_CC(paths, names):
    ## imports CC data from all files in a folder
    data = []
    for i, path in enumerate(paths):
        for file in os.listdir(path):
            if file.endswith('Z2'):
                vols, numbers = import_data_CC(os.path.join(path, file))
                data.append([[path, names[i], file], vols, numbers])
    #print(data)
    return data

def calcbarsize(data):
    return max(data[1])/len(data[1])

def norm_data_cc(data, CC_norm_data):
    if CC_norm_data:
        for entry in data:
            total = sum(entry[2])
            for i, number in enumerate(entry[2]):
                entry[2][i] = number / total
    return data
                
def edit_label_CC(label, culture_name):
    ## edits CC label)
    label = label.rstrip("_") #idk if this is necessary always, maybe I just did some mistakes when saving my og data set
    label = label.replace(culture_name + '_', '') ##### Change this according to your naming scheme
    label = label.replace('_', '.')
    return label

def match_name(name, entry, listforculture):
    ### matches name of culture to data, and stripping all the unnecessary stuff to only give the horm conc afte
    if type(entry) != float:
        if re.match('^.*=?('+name+')', entry):
            entry = entry.lstrip(name)
            entry = entry.lstrip("_")
            entry = entry.rstrip("nM_2")
            entry = entry.replace("_", ".")
            entry = float(entry)
            listforculture.append(entry)
    return listforculture


def plot_CC(entry, fig=None, ax=None):
    if not fig or not ax:
        fig, ax = plt.subplots()
    ## plots CC data
    ax.bar(entry[1], entry[2], color="blue", alpha=0.7, width=calcbarsize(entry))
    return fig, ax

def plot_together_CC(data, culture_name, fig=None, ax=None):
    ## plots CC data together
    if not fig or not ax:
        fig, ax = plt.subplots()
    for entry in data:
        if re.match('^.*=?('+culture_name+')', entry[0]):
            label = edit_label_CC(entry[0], culture_name)
            ax.scatter(entry[1], entry[2], label=label, alpha=0.7, size=3)
    return fig, ax

def plot_together_CC_fit(data, CC_culm, result_master_fit, culture_name, fig=None, ax=None):
    if not fig or not ax:
        fig, ax = plt.subplots()
    colorcount = 0
    for entry in data:
        if re.match('^.*=?('+culture_name+')', entry[0]):
            colorcount += 1
    colors = getcolormap(colorcount)
    i = 0
    for j, entry in enumerate(data):
        if re.match('^.*=?('+culture_name+')', entry[0]):
            label = edit_label_CC(entry[0], culture_name)
            ax.scatter(entry[1], entry[2], label=label, alpha=0.7, s=3, color=colors[i])
            for index in [sublist[3] for sublist in result_master_fit]:
                if index == entry[3]:
                    pltfit(ax, entry[1], CC_culm, result_master_fit[index][1], result_master_fit[index][2], label=label, color=colors[i])
            i += 1
    return fig, ax

def gaus(X, C, X_mean, sigma):
    return C*exp(-(X-X_mean)**2/(2*sigma**2))

def gauslist(xlist, C, X_mean, sigma):
    ylist = []
    for X in xlist:
        ylist.append(gaus(X, C, X_mean, sigma))
    return ylist

def gauslistcum(xlist, C, X_mean, sigma):
    ylist = []
    for X in xlist:
        ylist.append(gaus(X, C, X_mean, sigma))
    for i, _ in enumerate(ylist[1:], start=1):
        ylist[i] = ylist[i] + ylist[i-1]
    return ylist

def errorgauslist(xlist, param_optimised, param_covariance_matrix, lowerupper, cummu):
    ylists = []
    for pm1 in [operator.pos, operator.neg]:
        for pm2 in [operator.pos, operator.neg]:
            for pm3 in [operator.pos, operator.neg]:
                if cummu != True:
                    ylists.append(gauslist(xlist, param_optimised[0] + pm1(param_covariance_matrix[0,0]), param_optimised[1] + pm2(param_covariance_matrix[1,1]), param_optimised[2] + pm3(param_covariance_matrix[2,2])))
                elif cummu == True:
                    ylists.append(gauslistcum(xlist, param_optimised[0] + pm1(param_covariance_matrix[0,0]), param_optimised[1] + pm2(param_covariance_matrix[1,1]), param_optimised[2] + pm3(param_covariance_matrix[2,2])))
    ylist_master = []
    ylists = list(zip(*ylists))
    for ylistentry in ylists:
        if lowerupper == 'lower':
            ylist_master.append(min(ylistentry))
        elif lowerupper == 'upper':
            ylist_master.append(max(ylistentry))
    return ylist_master

def fit(x, y, what):
    mean = np.average(x, weights=y)               
    sigma = np.sqrt(np.average((x - mean)**2, weights=y))
    C = max(y)
    print(sigma, mean, C)
    param_optimised, param_covariance_matrix = curve_fit(gaus,x,y,p0=[C,mean,sigma])
    #print fit Gaussian parameters
    print("\nFit parameters of " + what[2] + " from exp. " + what[1] + ": ")
    print("C = ", param_optimised[0], "+-",np.sqrt(param_covariance_matrix[0,0]))
    print("X_mean =", param_optimised[1], "+-",np.sqrt(param_covariance_matrix[1,1]))
    print("sigma = ", param_optimised[2], "+-",np.sqrt(param_covariance_matrix[2,2]))

    return param_optimised, param_covariance_matrix

def pltfit(ax, x, CC_culm, param_optimised, param_covariance_matrix, label="", color="orange"):
    if label != "":
        label = label + " fit"
    else:
        label = "Fit"
    if CC_culm != True:
        ax.plot(x, gauslist(x, param_optimised[0], param_optimised[1], param_optimised[2]), label=label, color=color)
        y_upper = errorgauslist(x, param_optimised, param_covariance_matrix, 'upper', False)
        y_lower = errorgauslist(x, param_optimised, param_covariance_matrix, 'lower', False)
        ax.fill_between(x, y_upper, y_lower, color=color, alpha=0.3)
    else:
        ax.plot(x, gauslistcum(x, param_optimised[0], param_optimised[1], param_optimised[2]), label=label, color=color)
        y_upper = errorgauslist(x, param_optimised, param_covariance_matrix, 'upper', True)
        y_lower = errorgauslist(x, param_optimised, param_covariance_matrix, 'lower', True)
        ax.fill_between(x, y_upper, y_lower, color="orange", alpha=0.3)
    return ax


def savexlsxfit_CC(result_master_ext, CC_path, CC_exp_name):
    ###
    result_master_int = copy.deepcopy(result_master_ext)
    #print(result_master_int)
    for i, _ in enumerate(result_master_int):
        for j in range(1, 4):
            result_master_int[i][j] = str(result_master_int[i][j][0]) + "+-" + str(result_master_int[i][j][1])
    #result_master_sorted = []
    #for name in culture_names:
    #    listforculture = []
    #    #print(result_master_int)
    #    for entry in result_master_int:
    #        listforculture = match_name(name, entry[0][2], listforculture)
    #    sorted_list = []
    #    sorted_list2 = []
    #    print(listforculture)
    #    sorted_list = sorted(listforculture, key=lambda x: x[0])
    #    for entry in sorted_list:
    #        entry[0] = name + "_" + str(entry[0]) + "nM"
    #        sorted_list2.append(entry)
    #    result_master_sorted += sorted_list2
    result_master = pd.DataFrame(result_master_int)
    result_master.columns = ['name', 'C', 'X_mean', 'sigma']
    result_master, unique_entries = sorting_dataframe(result_master, split_name_label=True, create_beaty=True)
    func_column = ["C*exp(-(X-X_mean)**2/(2*sigma**2))"] + [None]*(len(result_master_int)-1) ###change this to the function from gaus()
    
    result_master['function'] = func_column
    saveexcel(result_master, os.path.join(CC_path, CC_exp_name) + '_fit.xlsx')


### main
def plotfitdata():
    CC_paths, CC_exp_names, culture_names, custom_order, CC_norm_data, CC_culm, CC_fit, savepath, exp_name_master = importconfigCC()
    CC_norm_data = True
    data = import_all_data_CC(CC_paths, CC_exp_names)
    data = norm_data_cc(data, CC_norm_data)
    result_master = []
    for entry in data:
        entry[0][2] = entry[0][2].rstrip("_.=#Z2")
        param_optimised, param_covariance_matrix = fit(entry[1], entry[2], entry[0])
        result = [entry[0][2]]
        for i in range(3):
            result.append([param_optimised[i], param_covariance_matrix[i,i]])
        result_master.append(result)
    savexlsxfit_CC(result_master, savepath, exp_name_master)
    result_master = pd.DataFrame(result_master)
    result_master.columns = ['name', 'C', 'X_mean', 'sigma']
    result_master, unique_entries = sorting_dataframe(result_master, split_name_label=True, create_beaty=True)
    #printl(unique_entries)
    fig2, ax2 = plt.subplots()
    for name, start, leng in unique_entries:
        fig, ax = plt.subplots()
        endpoint = start + leng
        listforculture = result_master.iloc[start:endpoint,:].values.tolist()
        ax.scatter(x=[sublist[1] for sublist in listforculture], y=[sublist[3][0] for sublist in listforculture])
        #print([sublist[2][0] for sublist in listforculture])
        #ax.errorbar(x=[sublist[0] for sublist in listforculture], y=[sublist[2][0] for sublist in listforculture], yerr=[sublist[3][0] for sublist in listforculture], fmt='none', capsize=4)
        #print([sublist[3][0] for sublist in listforculture])
        ax.grid(True)
        ax.set_title(name+ " Fit results")
        ax.set_ylabel('Volume (fL)')
        ax.set_xlabel("Hormone concentration (nM)")
        fig.canvas.manager.set_window_title(exp_name_master + '_CellSize_' + name)
        save_path = os.path.join(savepath, exp_name_master) + '_CellSize_' + name + '.png'
        fig.savefig(save_path)
        print('Saved plot to ' + save_path)
        ax2.scatter(x=[sublist[1] for sublist in listforculture], y=[sublist[3][0] for sublist in listforculture], label=name)
    ax2.grid(True)
    ax2.set_title("Fit")
    ax2.set_ylabel('Volume (fL)')
    ax2.set_xlabel("Hormone concentration (nM)")
    ax2.legend()
    fig2.canvas.manager.set_window_title(exp_name_master + '_CellSize')
    save_path = os.path.join(savepath, exp_name_master) + '_CellSize.png'
    fig2.savefig(save_path)
    print('Saved plot to ' + save_path)

    plt.show()

def boxplot():
    CC_paths, CC_exp_names, culture_names, custom_order, CC_norm_data, CC_culm, CC_fit, savepath, exp_name_master = importconfigCC()
    data = import_all_data_CC(CC_paths, CC_exp_names)
    #data = norm_data_cc(data, CC_norm_data)
    for entry in data:
        entry[0] = entry[0][2].rstrip(".=#Z2")
    data = pd.DataFrame(data)
    data.columns = ['name', 'vols', 'numbers']
    data, unique_entries, unique_entries_detailed = sorting_dataframe(data, split_name_label=True, create_beaty=True, precise_unique=True)
    #printl(data)
    #printl(data['numbers'][1])
    data_weighted_master = []
    for detailname, start, leng in unique_entries_detailed:
        label = detailname.split("$&$", 1)[1]
        label = float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", label)[0])
        name = data['name'][start]
        expanded_data = []
        end = start + leng
        for i in range(start, end, 1):
            for j, _ in enumerate(data['numbers'][i]):
                expanded_data += [data['vols'][i][j]] * data['numbers'][i][j]
        data_weighted_master.append([name, label, expanded_data])
    #printl(data, pdnomax=True)
    for name, start, leng in unique_entries:
        cult_list = []
        for entry in data_weighted_master:
            if entry[0] == name:
                cult_list.append(entry)
        fig, ax = plt.subplots()
        ax.boxplot([sublist[2] for sublist in cult_list], positions=[sublist[1] for sublist in cult_list], showfliers=False)
        ax.grid(True)
        ax.set_ylabel('Volume (fL)')
        ax.set_xlabel("Hormone concentration (nM)")
        ax.set_title(name+ " Cell Size")
        fig.canvas.manager.set_window_title(exp_name_master + '_CellSize_Boxplot_' + name)
        save_path = os.path.join(savepath, exp_name_master + '_CellSize_Boxplot_' + name + '.png')
        fig.savefig(save_path)
        print('Saved plot to ' + save_path)
    plt.show()

def coultercounter():
    ## creates CC plots for each exp separately cumulatively
    CC_paths, CC_exp_names, culture_names, custom_order, CC_norm_data, CC_culm, CC_fit, savepath, exp_name_master = importconfigCC()
    data = import_all_data_CC(CC_paths, CC_exp_names)
    data = norm_data_cc(data, CC_norm_data)
    result_master = []
    for entry in data:
        entry[0] = entry[0].rstrip(".=#Z2")
        if CC_fit == True:
            param_optimised, param_covariance_matrix = fit(entry[1], entry[2], entry[0])
            result = [entry[0]]
            for i in range(3):
                result.append([param_optimised[i], param_covariance_matrix[i,i]])
            result_master.append(result)
        if CC_culm == True:
            for i, _ in enumerate(entry[2][1:], start=1):
                entry[2][i] = entry[2][i] + entry[2][i-1]
        fig, ax = plot_CC(entry)
        if CC_fit == True:
            ax = pltfit(ax, entry[1], CC_culm, param_optimised, param_covariance_matrix)
            
        ax.set_title(entry[0])
        ax.set_xlabel('Volume (fL)')
        if CC_norm_data == True:
            ax.set_ylabel('Fraction of cells')
        else:
            ax.set_ylabel('Number of cells')
        ax.grid(True)
        fig.canvas.manager.set_window_title(CC_exp_name + '_' + entry[0])
        save_path = os.path.join(CC_path, CC_exp_name) + '_' + entry[0] + '.png'
        fig.savefig(save_path)
        print('Saved plot to ' + save_path)
    if CC_fit == True:
        savexlsxfit_CC(result_master, CC_path, CC_exp_name, culture_names)
    plt.show()

def coulterocunter_together():
    ## creates CC plots for cultures together cumulatively
    CC_paths, CC_exp_names, culture_names, custom_order, CC_norm_data, CC_culm, CC_fit, savepath, exp_name_master = importconfigCC()
    data = import_all_data_CC(CC_paths, CC_exp_names)
    data = norm_data_cc(data, CC_norm_data)
    result_master_excel = []
    result_master_fit = []
    for i, entry in enumerate(data):
        data[i][0] = entry[0].rstrip(".=#Z2")
        data[i] = [entry[0], entry[1], entry[2], i]
        if CC_fit == True:
            param_optimised, param_covariance_matrix = fit(entry[1], entry[2], entry[0])
            result = [entry[0]]
            for j in range(3):
                result.append([param_optimised[j], param_covariance_matrix[j,j]])
            result_master_excel.append(result)
            result_plot = [entry[0]]
            result_plot.append(param_optimised)
            result_plot.append(param_covariance_matrix)
            result_plot.append(i)
            result_master_fit.append(result_plot)

    if CC_culm == True:
        for entry in data:
            for i, _ in enumerate(entry[2][1:], start=1):
                entry[2][i] = entry[2][i] + entry[2][i-1]
    for culture_name in culture_names:
        fig, ax = plt.subplots()
        if CC_fit != True:
            fig, ax = plot_together_CC(data, culture_name, fig, ax)
            ax.legend()
            ax = sort_labels_CC(ax, custom_order)
        else:
            fig, ax = plot_together_CC_fit(data, CC_culm, result_master_fit, culture_name, fig, ax)
            new_order = []
            for label in custom_order:
                new_order.append(label)
                new_order.append(label + ' fit')
            custom_order = new_order
            ax.legend()
            ax = labelreorg(ax, custom_order, deldouble=False)
        ax.set_title(culture_name)
        ax.set_xlabel('Volume (fL)')
        if CC_norm_data == True:
            ax.set_ylabel('Fraction of cells')
        else:
            ax.set_ylabel('Number of cells')
        ax.grid(True)
        fig.canvas.manager.set_window_title(CC_exp_name + '_' + culture_name)
        save_path = os.path.join(CC_path, CC_exp_name) + '_' + culture_name + '.png'
        fig.savefig(save_path)
        print('Saved plot to ' + save_path)
    if CC_fit == True:
        savexlsxfit_CC(result_master_excel, CC_path, CC_exp_name, culture_names)
    plt.show()
