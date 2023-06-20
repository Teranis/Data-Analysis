import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.cm as mcolors
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

def labelereorg(axs, custom_order=[], deldouble=True):
    axs = sort_labels(axs, custom_order)
    if deldouble:
        handles, labels = axs.get_legend_handles_labels()
        new_handles, new_labels = [], []
        colormap = []
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

###I WILL SUBJUGATE MATPLOTLIB