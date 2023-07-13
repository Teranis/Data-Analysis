import numpy as np
import pandas as pd
import os
from configload import importconfigspotMAX
import regex as re
from core import loadcsv, printl, labelreorg
import matplotlib as plt
###Small functions
def finding_xlsx(root_path, folder_filter, spotMAX_foldername, spotMAX_filename):
    base_xlsx_files_paths =[]
    xlsx_files_paths = []
    folder_list = os.listdir(root_path)
    folder_list = [os.path.join(root_path, folder_name, spotMAX_foldername) for folder_name in folder_list if folder_name.lower().startswith(folder_filter.lower())]
    #printl(folder_list)
    for folder_name in folder_list:
        try:
            folder_cont = os.listdir(folder_name)
        except:
            print("\nDid not find a spotMAX analysis for:\n" + folder_name)
            continue
        for file_name in folder_cont:
            if re.match(spotMAX_filename, file_name):
                base_xlsx_files_paths.append(os.path.join(folder_name, file_name))
                xlsx_files_paths.append(folder_name)
    return base_xlsx_files_paths, xlsx_files_paths


def group_by_frame(data):
    per_frame = []
    lastframe = data["frame_i"].iloc()[-1]
    frame_list = list(range(lastframe))
    #print(frame_list)
    for frame in frame_list:
        single_frame_list = []
        for line in data.values.tolist():
            if line[0] == frame:
                single_frame_list.append(line)
        per_frame.append(single_frame_list)
    return per_frame

def aggregate_mother_bud(per_frame):
    #printl(per_frame)
    mother_bud_match = []
    master_matches = []
    for frame in per_frame[:10]:
        matches = 0
        frame_mother_bud_matches = [frame[0][0]]
        used_IDs = []
        for cell in frame:
            #print(used_IDs)
            if cell[1] not in used_IDs:
                used_IDs.append(cell[1])
                #print(i, used_IDs)
                has_partner = False
                #print(cell)
                if cell[5] == -1:
                    cell.append(False)
                    cell = tuple(cell)
                    frame_mother_bud_matches.append((cell,))
                else:
                    for cell2 in frame:
                        if cell2[1] not in used_IDs:
                            if cell[5] == cell2[1]:
                                if cell[6] == 'mother':
                                    cell.append('mother') 
                                    cell2.append('bud')
                                    cell = tuple(cell)
                                    cell2 = tuple(cell2)
                                    frame_mother_bud_matches.append((cell, cell2))
                                elif cell[6] == 'bud':
                                    cell.append('bud')
                                    cell2.append('mother')
                                    cell = tuple(cell)
                                    cell2 = tuple(cell2)
                                    frame_mother_bud_matches.append((cell2, cell))
                                else:
                                    print('I am confusion! Bud mother wtf?\nFrame: ' + frame + "CellIDs: " + cell[1] + cell2[1])
                                    exit()
                                used_IDs.append(cell2[1])
                                matches += 1
                                has_partner = True
                                break
                    if has_partner == False:
                        cell.append(False)
                        cell = tuple(cell)
                        frame_mother_bud_matches.append((cell,))
                        used_IDs.append(cell[1])
        master_matches.append(matches)
        mother_bud_match.append(frame_mother_bud_matches)
    return {'mother_bud_match':mother_bud_match, 'master_matches':master_matches}

def boxplot(data, x_text=None, y_text=None, title=None, savepath=None, x_labels=None, y_labels=None, weights=None, labels=None, notch=None, sym=None, vert=None, whis=None, positions=None, widths=None, patch_artist=None, bootstrap=None, usermedians=None, conf_intervals=None, meanline=None, showmeans=None, showcaps=None, showbox=None, showfliers=None, boxprops=None, flierprops=None, medianprops=None, meanprops=None, capprops=None, whiskerprops=None, manage_ticks=True, autorange=False, zorder=None, capwidths=None, *args):
    mplargs = (notch, sym, vert, whis, patch_artist, bootstrap, usermedians, conf_intervals, meanline, showmeans, showcaps, showbox, showfliers, boxprops, flierprops, medianprops, meanprops, capprops, whiskerprops, manage_ticks, autorange, zorder, capwidths, *args)
    if type(data[0]) != list:
        if weights != None:
            data_exp = []
            for i, datap in enumerate(data):
                data_exp += [datap]*weights[i]
            data = data_exp

        if y_labels!=None:
            is_numberylabel = True 
            for label in labels:
                if not isinstance(label, float):
                    is_numberylabel = False

        fig, ax = plt.subplots()
        if is_numberylabel == True:
            ax.boxplot(data, positions=y_labels, labels=labels, widths=widths, *mplargs)
        else:
            labels = y_labels
            ax.boxplot(data, labels=[labels], widths=widths, *mplargs)
    else:
        if weights != None:
            data_exp = []
            for j, datalist in data:
                datalist_exp = []
                for i, datap in enumerate(datalist):
                    datalist_exp += [datap]*weights[j][i]
                data_exp.append(datalist_exp)
            data = data_exp
        if x_labels != None:
            groupsize = len(data)/len(x_labels)
            width = 0.75/(groupsize+1)
            width_tot = 1/(groupsize+1)
            offset = 0
            mult = 0
            for i, label in enumerate(x_labels):
                startp = i*groupsize
                endp = (i+1)*groupsize
                pos = [i+j*width_tot for j in range(0, groupsize, 1)]
                ax.boxplot(data[startp:endp], widths=width, labels=labels, positions=pos, *mplargs)
                ax.set_xticklabels(x_labels)
        else:
            ax.boxplot(data, widths=widths, labels=labels, positions=pos, *mplargs)
    ax.grid(True)
    if y_text != None:
        ax.set_ylabel(y_text)
    if x_text != None:
        ax.set_xlabel(x_text)
    if title != None:
        ax.set_title(title)
    if labels != None:
        if y_labels!=None:
            is_numberlabel = True 
            for label in labels:
                if not isinstance(label, float):
                    is_numberlabel = False
        ax.legend()
        ax = labelreorg(ax, find_custom_order=is_numberlabel)
    title = title.replace(".", "_").replace(" ", "_")
    fig.canvas.manager.set_window_title(title)
    save_path = os.path.join(savepath, title)
    fig.savefig(save_path)
    print('Saved box-plot to ' + save_path)

###Main
def boxplot():
    config = importconfigspotMAX()
    paths = config["path"]
    folder_filter = config["folder_filter"]
    spotMAX_foldername = config["spotMAX_foldername"]
    spotMAX_filename = config["spotMAX_filename"]
    file_master = []
    last_edited_frame = []
    for path in paths:
        base_xlsx_files_paths, xlsx_files_paths = finding_xlsx(path, folder_filter, spotMAX_foldername, spotMAX_filename)
        #printl(base_xlsx_files_paths, xlsx_files_paths)
        for file_path in base_xlsx_files_paths:
            #printl(file_path, pretty=True)
            data = loadcsv(file_path)
            per_frame = group_by_frame(data)
            last_edited_frame.append(per_frame[-1][0])
            aggregate_mother_bud_res = aggregate_mother_bud(per_frame)
            mother_bud_match = aggregate_mother_bud_res['mother_bud_match']
            printl(mother_bud_match, pretty=True)

            for frame in mother_bud_match:
                mother_bud_aggr_frame = {
                    'Cell_Size':[],
                    'Mito_Size':[],
                    'Rel_Size':[]
                    }
                mother_frame = {
                    'Cell_Size':[],
                    'Mito_Size':[],
                    'Rel_Size':[]
                    }
                bud_frame = {
                    'Cell_Size':[],
                    'Mito_Size':[],
                    'Rel_Size':[]
                    }
                single_frame = {
                    'Cell_Size':[],
                    'Mito_Size':[],
                    'Rel_Size':[]
                    }
                for match in frame[1:]:
                    if match[0][-1] == False:
                        single_frame['Cell_Size'].append(match[0][12])
                        single_frame['Mito_Size'].append(match[0][15])
                        single_frame['Rel_Size'].append(match[0][15]/match[12])
                    else:
                        mother_frame['Cell_Size'].append(match[0][12])
                        mother_frame['Mito_Size'].append(match[0][15])
                        mother_frame['Rel_Size'].append(match[0][15]/match[12])
                        bud_frame['Cell_Size'].append(match[1][12])
                        bud_frame['Mito_Size'].append(match[1][15])
                        bud_frame['Rel_Size'].append(match[1][15]/match[12])
                        mother_bud_aggr_frame['Cell_Size'].append(match[0][12]+match[1][12])
                        mother_bud_aggr_frame['Mito_Size'].append(match[0][15]+match[1][15])
                        mother_bud_aggr_frame['Rel_Size'].append((match[0][15]+match[1][15])/(match[0][12]+match[1][12]))



                for key, value in mother_bud_aggr_frame.items():
                    avg = sum(value)/len(value)
                    mother_bud_aggr_frame[key] = avg

                for key, value in mother_frame.items():
                    avg = sum(value)/len(value)
                    mother_frame[key] = avg

                for key, value in bud_frame.items():
                    avg = sum(value)/len(value)
                    bud_frame[key] = avg

                for key, value in single_frame.items():
                    avg = sum(value)/len(value)
                    single_frame[key] = avg
                    

                single['Cell_Size'].append(single_frame['Cell_Size'])
                single['Mito_Size'].append(single_frame['Mito_Size'])
                single['Rel_Size'].append(single_frame['Rel_Size'])
                mother['Cell_Size'].append(mother_frame['Cell_Size'])
                mother['Mito_Size'].append(mother_frame['Mito_Size'])
                mother['Rel_Size'].append(mother_frame['Rel_Size'])
                bud['Cell_Size'].append(bud_frame['Cell_Size'])
                bud['Mito_Size'].append(bud_frame['Mito_Size'])
                bud['Rel_Size'].append(bud_frame['Rel_Size'])
                mother_bud_aggr['Cell_Size'].append(mother_bud_aggr_frame['Cell_Size'])
                mother_bud_aggr['Mito_Size'].append(mother_bud_aggr_frame['Mito_Size'])
                mother_bud_aggr['Rel_Size'].append(mother_bud_aggr_frame['Rel_Size'])
            file_master.append([file_path, single, mother, bud, mother_bud_aggr])
    print('Using data untill frame: ' + last_edited_frame)
    for entry in file_master:
        for dicti in entry[1:]:

