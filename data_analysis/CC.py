from calendar import c
import os
import matplotlib.pyplot as plt
import regex as re
import math
from configload import importconfigCC
from core import sort_labels as sort_labels_CC
from scipy import stats
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import numpy as np
import pandas as pd
import operator
from core import saveexcel, labelreorg
from core import getcolormap
import copy
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

#def sort_labels_CC(ax, custom_order):
#    ## sorts out CC labels
#    if custom_order != []:
#        handles, labels = plt.gca().get_legend_handles_labels()
#        sort_list = sorted(range(len(labels)), key=lambda k: custom_order.index(labels[k]))
#        ax.legend([handles[idx] for idx in sort_list],[labels[idx] for idx in sort_list])
#    return ax

def plot_CC(entry, fig=None, ax=None):
    if not fig or not ax:
        fig, ax = plt.subplots()
    ## plots CC data
    ax.bar(entry[1], entry[2], color="blue", alpha=0.7)
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
    mean = sum(np.multiply(x, y))/sum(y)                  
    sigma = sum(np.power(np.multiply(y, (np.subtract(x, mean))) ,2))/sum(y)
    param_optimised, param_covariance_matrix = curve_fit(gaus,x,y,p0=[max(y),mean,sigma],maxfev=5000)
    #print fit Gaussian parameters
    print("\nFit parameters of " + what + ": ")
    print("C = ", param_optimised[0], "+-",np.sqrt(param_covariance_matrix[0,0]))
    print("X_mean =", param_optimised[1], "+-",np.sqrt(param_covariance_matrix[1,1]))
    print("sigma = ", param_optimised[2], "+-",np.sqrt(param_covariance_matrix[2,2]))

    return param_optimised, param_covariance_matrix

def pltfit(ax, x, CC_culm, param_optimised, param_covariance_matrix, label="", color="orange"):
    if label != "":
        label = label + " fit"
    else:
        label = "fit"
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


def savexlsxfit_CC(result_master_ext, CC_path, CC_exp_name, culture_names, custom_order=[]):
    ###
    result_master = copy.deepcopy(result_master_ext)
    for i, _ in enumerate(result_master):
        for j in range(1, 4):
            result_master[i][j] = str(result_master[i][j][0]) + "+-" + str(result_master[i][j][1])
    result_master_sorted = []
    for name in culture_names:
        listforculture = []
        for entry in result_master:
            if type(entry[0]) != float:
                if re.match('^.*=?('+name+')', entry[0]):
                    entry[0] = entry[0].lstrip(name)
                    entry[0] = entry[0].lstrip("_")
                    entry[0] = entry[0].rstrip("nM_2")
                    entry[0] = entry[0].replace("_", ".")
                    entry[0] = float(entry[0])
                    listforculture.append(entry)
        sorted_list = []
        sorted_list2 = []
        if custom_order == []:
            sorted_list = sorted(listforculture, key=lambda x: x[0])
        else:
            custom_order = [entry.rstrip("nM") for entry in custom_order]
            for entry in custom_order:
                for entry2 in listforculture:
                    if entry == entry2[0]:
                        sorted_list.append(listforculture)

        for entry in sorted_list:
            entry[0] = name + "_" + str(entry[0]) + "nM"
            sorted_list2.append(entry)
        result_master_sorted += sorted_list2
    ###
    func_column = ["C*exp(-(X-X_mean)**2/(2*sigma**2))"] + [None]*(len(result_master_sorted)-1) ###change this to the function from gaus()
    result_master = pd.DataFrame(result_master_sorted)
    result_master['function'] = func_column
    result_master.columns = ['name', 'C', 'X_mean', 'sigma', 'function']
    saveexcel(result_master, os.path.join(CC_path, CC_exp_name) + '_fit.xlsx')
### main
def plotfitdata():
    CC_path, CC_exp_name, culture_names, custom_order, CC_norm_data, CC_culm, CC_fit = importconfigCC()
    data = import_all_data_CC(CC_path)
    data = norm_data_cc(data, CC_norm_data)
    result_master = []
    for entry in data:
        entry[0] = entry[0].rstrip(".=#Z2")
        param_optimised, param_covariance_matrix = fit(entry[1], entry[2], entry[0])
        result = [entry[0]]
        for i in range(3):
            result.append([param_optimised[i], param_covariance_matrix[i,i]])
        result_master.append(result)
    savexlsxfit_CC(result_master, CC_path, CC_exp_name, culture_names)
    ihatehowlistswork = []
    for name in culture_names:
        listforculture = []
        for entry in result_master:
            if type(entry[0]) != float:
                if re.match('^.*=?('+name+')', entry[0]):
                    entry[0] = entry[0].lstrip(name)
                    entry[0] = entry[0].lstrip("_")
                    entry[0] = entry[0].rstrip("nM_2")
                    entry[0] = entry[0].replace("_", ".")
                    entry[0] = float(entry[0])
                    listforculture.append(entry)

        fig, ax = plt.subplots()
        ax.scatter(x=[sublist[0] for sublist in listforculture], y=[sublist[2][0] for sublist in listforculture])
        ax.grid(True)
        ax.set_title(name+ " fit")
        ax.set_ylabel('Volume (uL)')
        ax.set_xlabel("Hormone concentration (nM)")
        fig.canvas.manager.set_window_title(CC_exp_name + '_CellSize_' + name)
        save_path = os.path.join(CC_path, CC_exp_name) + '_CellSize_' + name + '.png'
        plt.savefig(save_path)
        print('Saved plot to ' + save_path)
        listforculture = [str(name)] + listforculture
        ihatehowlistswork.append(listforculture)

    fig, ax = plt.subplots()
    colors = getcolormap(len(ihatehowlistswork))
    
    for listforculture in ihatehowlistswork:
        label = listforculture[0]
        del listforculture[0]
        print([sublist[1] for sublist in listforculture])
        print([sublist[2][0] for sublist in listforculture])
        ax.scatter(x=[sublist[0] for sublist in listforculture], y=[sublist[2][0] for sublist in listforculture], label=label)
    ax.grid(True)
    ax.set_title(" fit")
    ax.set_ylabel('Volume (uL)')
    ax.set_xlabel("Hormone concentration (nM)")
    ax.legend()
    fig.canvas.manager.set_window_title(CC_exp_name + '_CellSize')
    save_path = os.path.join(CC_path, CC_exp_name) + '_CellSize.png'
    plt.savefig(save_path)
    print('Saved plot to ' + save_path)


    plt.show()

def coultercounter():
    ## creates CC plots for each exp separately cumulatively
    CC_path, CC_exp_name, culture_names, custom_order, CC_norm_data, CC_culm, CC_fit = importconfigCC()
    data = import_all_data_CC(CC_path)
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
        ax.set_xlabel('Volume (uL)')
        if CC_norm_data == True:
            ax.set_ylabel('Fraction of cells')
        else:
            ax.set_ylabel('Number of cells')
        ax.grid(True)
        fig.canvas.manager.set_window_title(CC_exp_name + '_' + entry[0])
        save_path = os.path.join(CC_path, CC_exp_name) + '_' + entry[0] + '.png'
        plt.savefig(save_path)
        print('Saved plot to ' + save_path)
    if CC_fit == True:
        savexlsxfit_CC(result_master, CC_path, CC_exp_name, culture_names, custom_order)
    plt.show()

def coulterocunter_together():
    ## creates CC plots for cultures together cumulatively
    CC_path, CC_exp_name, culture_names, custom_order, CC_norm_data, CC_culm, CC_fit = importconfigCC()
    data = import_all_data_CC(CC_path)
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
        ax.set_xlabel('Volume (uL)')
        if CC_norm_data == True:
            ax.set_ylabel('Fraction of cells')
        else:
            ax.set_ylabel('Number of cells')
        ax.grid(True)
        fig.canvas.manager.set_window_title(CC_exp_name + '_' + culture_name)
        save_path = os.path.join(CC_path, CC_exp_name) + '_' + culture_name + '.png'
        plt.savefig(save_path)
        print('Saved plot to ' + save_path)
    savexlsxfit_CC(result_master_fit, CC_path, CC_exp_name, culture_names, custom_order)
    plt.show()