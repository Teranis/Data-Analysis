import os
import sys ###
import subprocess
import matplotlib.pyplot as plt
import matplotlib.cm as mcolors
import pandas as pd
import time
import math as m
import numpy as np
import itertools
import inspect ###
from datetime import datetime
from pprint import pprint
import regex as re
import concurrent.futures
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
    if custom_order:
        legend = ax.get_legend()
        #printl([text.get_text() for text in legend.get_texts()])
        #print(legend.get_lines())
        handles, labels = legend.legendHandles, [text.get_text() for text in legend.get_texts()]
        #printl(handles, labels)
        sort_list = sorted(range(len(labels)), key=lambda k: custom_order.index(labels[k]))
        ax.legend(handles=[], labels= [])
        ax.legend([handles[idx] for idx in sort_list],[labels[idx] for idx in sort_list])
    return ax

def labelreorg(axs, custom_order=[], deldouble=True, find_custom_order=False):
    if find_custom_order:
        labels = axs.get_legend_handles_labels()[1]
        for i, label in enumerate(labels):
            labels[i] = (float(re.findall(r"\d+\.?\d+", label)[0]), label)
        labels = sorted(labels, key=lambda x: x[0])
        custom_order = [label[1] for label in labels]
        if deldouble:
            custom_order_single = []
            for entry in custom_order:
                if entry not in custom_order_single:
                    custom_order_single.append(entry)
            custom_order = custom_order_single
        #print("Custom order: ")
        #print(custom_order)
    
    if deldouble:
        handles, labels = axs.get_legend_handles_labels()
        new_handles, new_labels = [], []
        for handle, label in zip(handles, labels):
            if label not in new_labels:
                new_handles.append(handle)
                new_labels.append(label)

        colors = getcolormap(len(new_labels))

        for obj in axs.get_children():
            #print(type(obj))
            #if obj.get_label() in new_labels:
                label = obj.get_label()
                #printl(obj.get_label(), type(obj))
                if str(label) in new_labels:
                #if isinstance(obj, (patches.Patch, lines.Line2D, collections.PathCollection, collections.PolyCollection)):
                     obj.set_color(colors[new_labels.index(label)])

        axs.legend(new_handles, new_labels)

    axs = sort_labels(axs, custom_order)
    return axs

def saveexcel(what, where):
    already_warned = False
    while True:
        try:
            what.to_excel(where, index=False)
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

def calcerrorslowerupper(func, x, *args, cum=False, step=1):
    '''
    Calculates the upper and lower error bounds for the fit parameters.
    syntax: calcerrorslowerupper(func, x, *args) where func is the function, x the value(s) for the first arg of func, and args are a list of the args IN THE RIGHT ORDER for func as follows: args = (value, uncertainty). Returns a tuple of the upper and lower bounds of the fit parameters. (max(ys), min(ys))
    '''
    #printl(type(x))
    typex = type(x)
    consts = []
    steps = [-i/step for i in range(step+1)]
    steps += [i/step for i in range(1, step+1, 1)]
    #print(steps)
    for arg in args:
        const = tuple([arg[0] + arg[1]*step for step in steps])
        consts.append(const)
    permutations = list(itertools.product(*consts))
    #printl(len(permutations))
    if typex == float or typex == int:
        #print("I RUN")
        if cum == True: 
            printl("I am very confused! Do you want to create a cum list out of a single x?")
            exit()
        ys = []
        #printl(permutations)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(func, x, *permutation) for permutation in permutations]
            ys = [future.result() for future in futures]
        return max(ys), min(ys)
    else:
        y_min, y_max = [], []
        for xentry in x:
            ys = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(func, xentry, *permutation) for permutation in permutations]
                ys = [future.result() for future in concurrent.futures.as_completed(futures)]
            #printl(permutations)
            y_min.append(min(ys))
            y_max.append(max(ys))
        if cum == True:
            for y_list in y_min, y_max:
                for i, _ in enumerate(y_list[1:], start=1):
                    y_list[i] = y_list[i] + y_list[i-1]
        return y_max, y_min

def create_uniques_list(name_list):
    name_unique = []
    for i, name in enumerate(name_list):
        if name not in [entry[0] for entry in name_unique]:
            name_unique.append([name, i, 1])
        else:
            for inx, name2 in enumerate([entry[0] for entry in name_unique]):
                if name == name2:
                    inx_match = inx
                    break
            name_unique[inx_match][2] = name_unique[inx_match][2] + 1
    return name_unique


def sorting_dataframe(data_frame, split_name_label=False, create_beaty=False, precise_unique=False):
    if split_name_label == True:
        singlename, labels = [], []
        for name in data_frame.iloc[:, 0].tolist():
            if create_beaty == True:
                name = name.lstrip().rstrip().lstrip("_").rstrip("_")
            name1, label1 = name.split('_', 1)
            if create_beaty == True:
                name1 = name1.lstrip().rstrip().lstrip("_").rstrip("_")
                label1 = label1.rstrip().lstrip("_").rstrip("_").replace("_", ".").replace("nM", " nM")
            labels.append(label1)
            singlename.append(name1)
        data_frame.insert(1, 'labels', labels)
        data_frame.iloc[:, 0] = singlename

    sorted_df = pd.DataFrame()
    data_frame = data_frame.sort_values(by=data_frame.columns[0])

    name_list = data_frame.iloc[:, 0].tolist()
    label_list = data_frame.iloc[:, 1].tolist()
    name_unique = create_uniques_list(name_list)
   
    for name in name_unique:
        endpoint = name[1] + name[2]
        sliced_df = data_frame.iloc[name[1]:endpoint].copy()
        label_list = sliced_df.iloc[:, 1].tolist()
        for i, label in enumerate(label_list):
            number = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", label)
            if len(number) == 1:
                label_list[i] = float(number[0])
            else:
                label_list[i] = label
        sliced_df['temp_column'] = label_list
        sliced_df = sliced_df.sort_values(by='temp_column')
        sliced_df = sliced_df.drop(columns='temp_column')
        sorted_df = pd.concat([sorted_df, sliced_df], axis=0)
    sorted_df = sorted_df.reset_index(drop=True)
    #pd.set_option('display.max_columns', 100)
    #print(sorted_df)
    if precise_unique == True:
        name_list = sorted_df.iloc[:, 0].tolist()
        label_list = sorted_df.iloc[:, 1].tolist()
        name_label_list = [(name, label) for name, label in zip(name_list, label_list)]
        namelabel_unique = create_uniques_list(name_label_list)
    if precise_unique == False:
        return sorted_df, name_unique
    elif precise_unique == True:
        return sorted_df, name_unique, namelabel_unique

def printl(*objects, pretty=False, is_decorator=False, **kwargs):
    # Copy current stdout, reset to default __stdout__ and then restore current
    current_stdout = sys.stdout
    sys.stdout = sys.__stdout__
    timestap = datetime.now().strftime('%H:%M:%S')
    currentframe = inspect.currentframe()
    outerframes = inspect.getouterframes(currentframe)
    idx = 2 if is_decorator else 1
    callingframe = outerframes[idx].frame
    callingframe_info = inspect.getframeinfo(callingframe)
    filpath = callingframe_info.filename
    filename = os.path.basename(filpath)
    print_func = pprint if pretty else print
    if "pdnomax" in kwargs:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        del kwargs["pdnomax"]
    if "pdrowmax" in kwargs:
        pd.set_option('display.max_rows', kwargs["pdrowmax"])
        del kwargs["pdrowmax"]
    if "pdcolmax" in kwargs:
        pd.set_option('display.max_columns', kwargs["pdcolmax"])
        del kwargs["pdcolmax"]
    print('*'*30)
    print(f'{timestap} - File "{filename}", line {callingframe_info.lineno}:')
    if 'sep' not in kwargs:
        kwargs['sep'] = ', '
    if pretty:
        del kwargs['sep']
    print_func(*objects, **kwargs)
    print('='*30)
    sys.stdout = current_stdout
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
###I WILL SUBJUGATE MATPLOTLIB