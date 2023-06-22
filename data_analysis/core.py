import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.cm as mcolors
import pandas as pd
import time
import math as m
import numpy as np
#Organization (use search with the appropriate number of # followed by a space)
##### Configs (Outsourced to config.json))
#### Sections: functions, main functions
### Subsections: OD, CC
## Single functions


#### Functions
### ConfigLoad

def openpath():
    file_dir = os.path.dirname(__file__)
    subprocess.run(['cd', file_dir], shell=True)

def openexp():
    path = os.path.dirname(__file__)
    subprocess.Popen(['explorer', path])

def openconfig():
    file_dir = os.path.dirname(__file__)
    for filename in ['configCC.json', 'configOD.json']:
        filename = os.path.join(file_dir, filename)
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
    file_dir = os.path.dirname(__file__)
    filename = os.path.join(file_dir, 'OD_measurements_example.xlsx')
    try:
        subprocess.run(['start', '', filename], shell=True)  # For Windows
    except FileNotFoundError:
        try:
            subprocess.run(['open', '-a', 'Microsoft Excel', filename])  # For macOS
        except FileNotFoundError:
            print("Unable to open the file with Excel.")

def sort_labels(ax, custom_order):
    ## sorts out CC labels
    if custom_order != []:
        handles, labels = ax.get_legend_handles_labels()
        sort_list = sorted(range(len(labels)), key=lambda k: custom_order.index(labels[k]))
        ax.legend([handles[idx] for idx in sort_list],[labels[idx] for idx in sort_list])
    return ax

def labelreorg(axs, custom_order=[], deldouble=True):
    axs = sort_labels(axs, custom_order)
    if deldouble:
        handles, labels = axs.get_legend_handles_labels()
        new_handles, new_labels = [], []
        for handle, label in zip(handles, labels):
            if label not in new_labels:
                new_handles.append(handle)
                new_labels.append(label)

        color_map = mcolors.get_cmap('tab10')
        colors = [color_map(i) for i in range(len(new_labels))]

        for obj, label in zip(axs.get_children(), labels):
            obj.set_color(colors[new_labels.index(label)])
            
        axs.legend(new_handles, new_labels)
    return axs

def saveexcel(what, where):
    already_warned = False
    while True:
        try:
            what.to_excel(where)
        except PermissionError:
            if not already_warned:
                print('PermissionError: Please close the excel file before saving.')
                already_warned = True
            time.sleep(0.1)
            continue
        print('Saved excel to', where)
        break

def loadexcel(where):
    already_warned = False
    while True:
        try:
            data = pd.read_excel(where)
        except PermissionError:
            if not already_warned:
                print('PermissionError: Please close the excel file before saving.')
                already_warned = True
            time.sleep(0.1)
        else:
            print('Loaded excel from', where)
            return data

def getcolormap(howmany):
    if howmany <= 10:
        mod = 10 % howmany
        step = (10 - mod) // howmany
        color_map = mcolors.get_cmap('tab10')
        colors = [color_map(i) for i in range(0, 10, step)]
    elif howmany <= 20:
        mod = 20 % howmany
        step = (20 - mod) // howmany
        color_map = mcolors.get_cmap('tab20')
        colors = [color_map(i) for i in range(0, 20, step)]
    else:
        color_map = mcolors.get_cmap()
        colors = [color_map(i) for i in np.linspace(0, 1, howmany)]
    return colors
###I WILL SUBJUGATE MATPLOTLIB