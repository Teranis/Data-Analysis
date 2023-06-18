import os
import matplotlib.pyplot as plt
import regex as re
import math
from configload import importconfigCC


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


### main
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