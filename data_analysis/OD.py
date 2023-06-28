from math import log, exp, sqrt
import os
from re import M
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.cm as mcolors
import statsmodels.api as sm
from configload import importconfigOD
from core import  labelreorg, saveexcel, getcolormap, calcerrorslowerupper, printl
from core import loadexcel as import_data_OD
import regex as re
import copy
#from data_analysis.configload import importconfigOD
#from data_analysis.core import  labelreorg, saveexcel, getcolormap
#From data_analysis.core import loadexcel as import_data_OD

### small

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
    #print(data)
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
        legend.append(str(data_metadata.iloc[i, 1]) + ' nM') #Legend Unit here
    #print(legend)
    return legend

def getmetadata_genstuff(data):
    ## getting metadata
    no_timepoints = data.iloc[0,:].notna().sum()
    total_pos = data.iloc[:,1].notna().sum()
    no_culture = data.iloc[:,0].notna().sum()
    no_perculture = int(total_pos / no_culture)
    #print(no_timepoints, total_pos, no_culture, no_perculture)
    return no_timepoints, total_pos, no_culture, no_perculture

def fit_curve(time, start_OD, D):
    ## exp fit
    OD = start_OD*np.exp(time*D)
    return OD

def calc_OD(time, DT, startOD):
     return 2**(time/DT) * startOD

def calc_OD_lin(time, D, startOD):
     return (2/D)*time + startOD

def fit_curve_lin(time, start_OD, D):
    ## linear fit
    OD = start_OD + time*D
    return OD

#def fitting(fit_curve, ODs, time, start_OD, fitstartval):
### fitting the data using scipy
#    D, cov_ma = curve_fit(lambda time, D: fit_curve(time, start_OD, D), time, ODs, fitstartval)
#    return D

def fitting_new(ODs, time, start_OD, fitstartval, OD_exp_fit, culture_name, legend):
    ## fitting the data using statmodels
    ODarray = np.array([])
    timearray = np.array([])
    ODs = np.array(ODs)
    for i, OD in enumerate(ODs):
        if np.isnan(OD) == False:
            ODarray = np.append(ODarray, OD)
            timearray = np.append(timearray, time[i])

    time = timearray
    ODs = ODarray
    if OD_exp_fit != True:
        fitstartvals = [0, fitstartval]
        for i, OD in enumerate(ODs):
            ODs[i] = OD/start_OD
        time = sm.add_constant(time)
        #print(ODs)
        #print(time)
        #print(fitstartval)
        #ODs[:, 0] = start_OD
        #print(ODs)
        model = sm.OLS(ODs, time)
        results = model.fit(start_params=fitstartvals)
    else:
        fitstartvals = [0, log(2)/fitstartval]
        #print(ODs)
        for i, OD in enumerate(ODs):
            ODs[i] = log(OD/start_OD)
        time = sm.add_constant(time)
        #print(ODs)
        #print(time)
        model = sm.OLS(ODs, time)
        results = model.fit()

    print("\n\n" + culture_name + legend)
    print(fitstartvals)
    print(results.summary())
    return results

def metadata_congdata(data_master):
    data_cong = []
    data_lengths = []
    data_times = []
    data_names = []
    data_legends = []
    data_no_cultures = []
    data_no_perculture = []
    data_no_timepoints = []
    data_total_pos = []
    for data in data_master:
        no_timepoints, total_pos, no_cultures, no_perculture = getmetadata_genstuff(data)
        #print(data)
        #data_cong += data
        data_cong = None
        data_lengths.append(len(data.columns))
        data_times.append(metadata_time(data, no_timepoints))
        data_names.append(metadata_names(data, total_pos, no_perculture))
        data_legends.append(metadata_legend(data, total_pos))
        data_no_cultures.append(no_cultures)
        data_no_perculture.append(no_perculture)
        data_no_timepoints.append(no_timepoints)
        data_total_pos.append(total_pos)
        #print(data_names, data_legends)
    return data_cong, data_lengths, data_times, data_names, data_legends, data_no_cultures, data_no_perculture, data_no_timepoints, data_total_pos

def match_names(data_names, starting_list_index=0, name_matches=[]):
    ##del already matched names
    data_names_copy = copy.deepcopy(data_names)
    #print(data_names_copy)
    for matches in name_matches:
        for match in matches:
            #print([matches[0][0],matches[0][1]])
            #print(data_names[matches[0][0]][matches[0][1]])
            index = data_names_copy[match[0]].index(data_names[matches[0][0]][matches[0][1]])
            data_names_copy[match[0]][index] = None
    #print(data_names_copy)
    for i, name1 in enumerate(data_names_copy[starting_list_index]):
        if name1 != None:
            name_match = [(starting_list_index, i)]
            for j, name_list in enumerate(data_names_copy[starting_list_index+1:]):
                j = starting_list_index + 1+ j
                for k, name2 in enumerate(name_list):
                    if re.match(name1, name2):
                        name_match.append((j, k))
            #print(data_names_copy)
            #print(name_match)
            name_match = tuple(name_match)
            name_matches.append(name_match)
    starting_list_index += 1
    if starting_list_index < len(data_names):
        #print(name_matches)
        return match_names(data_names, starting_list_index, name_matches)
    else:
        name_matches = tuple(name_matches)
        print(name_matches)
        return name_matches
    
def prepdata_data_multexp(data_master, OD_norm_data):
    data_cong, data_lengths, data_times, data_names, data_legends, data_no_cultures, data_no_perculture, data_no_timepoints, data_total_pos = metadata_congdata(data_master)

    data_names_legends = []
    for i, _ in enumerate(data_names):
        name_legends = []
        for j, name in enumerate(data_names[i]):
            for k in range(data_no_perculture[i]):
                name_legend = str(name) + "$" + data_legends[i][j*data_no_perculture[i]+k]
                name_legends.append(name_legend)
        data_names_legends.append(name_legends)    
    name_legend_matches = match_names(data_names_legends)

    for i, datap in enumerate(data_master):
        data_master[i] = cut_data(datap)
        if OD_norm_data == True:
            data_master[i] = norm_data(data_master[i], data_total_pos[i], data_no_timepoints[i])

    return data_cong, data_lengths, data_times, data_names, data_legends, data_no_cultures, data_no_perculture, data_no_timepoints, data_total_pos, name_legend_matches, data_master, data_names_legends

def find_cult_list(data_names):

    name_dict = {}
    for i, data_list in enumerate(data_names):
        #printl(data_list)
        for j, name in enumerate(data_list):
            og_name = name
            name.lower().lstrip().rstrip()
            if name in name_dict:
                name_dict[name].append((i, j))
            else:
                name_dict[name] = [og_name, (i, j)]
    culture_list = [value for key, value in name_dict.items()]
    return culture_list
### Main
def odplot():
    ## creates plots for each culture
    excel_paths, exp_name, OD_norm_data, use_fit, OD_exp_fit, OD_add_error_to_OD_plot, exp_names = importconfigOD()
    data_master = []
    for excel_path in excel_paths:
        print("\n\nData from "+ excel_path)
        data = import_data_OD(excel_path)
        data_master.append(data)
        print(data)
    for masterindex, datap_master in enumerate(data_master):
        data = datap_master.copy(deep=True)
        no_timepoints, total_pos, no_cultures, no_perculture = getmetadata_genstuff(data)
        times = metadata_time(data, no_timepoints)
        names = metadata_names(data, total_pos, no_perculture)
        legend = metadata_legend(data, total_pos)
        data = cut_data(data)
        if len(datap_master) > 1:
            exp_name = exp_names[masterindex].replace("_", ".")
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
            exp_name = exp_name.replace(".", "_")
            fig.canvas.manager.set_window_title(str(exp_name) + '_' + culturename)
            #print(os.path.join(os.path.dirname(excel_path), exp_name + '_' + culturename))
            plt.savefig(os.path.join(os.path.dirname(excel_path), str(exp_name) + '_' + culturename + '.png'))


    if len(data_master) > 1:
        data_cong, data_lengths, data_times, data_names, data_legends, data_no_cultures, data_no_perculture, data_no_timepoints, data_total_pos, name_legend_matches, data_master, data_names_legends = prepdata_data_multexp(data_master, OD_norm_data)
        for i, match in enumerate(name_legend_matches):
            fig, ax = plt.subplots()
            for listindex, entryindex in match:
                data = data_master[listindex]
                label = str(data_names_legends[listindex][entryindex].replace("$", " ")) + " from run: " + str(exp_names[listindex]).replace("_", ".")
                while len(data_times[listindex]) > len(data.iloc[entryindex]):
                    del data_times[listindex][-1]
                #print(data_times[listindex])
                #print(data)
                #print(label)
                ax.plot(data_times[listindex], data.iloc[entryindex], marker='o', label=label)
            ax.set_title(data_names_legends[listindex][entryindex].replace("$", " "))
            ax.set_xlabel('Time (h)')
            if OD_norm_data == True:
                ax.set_ylabel('Normalized Optical Density')
            else:
                ax.set_ylabel('Optical Density')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True)
            savename = data_names_legends[listindex][entryindex].replace("$", "_").replace(".", "_").replace(" ", "_")
            fig.canvas.manager.set_window_title(savename + '_all_runs')
            #print(os.path.join(os.path.dirname(excel_path), exp_name + '_' + culturename))
            plt.savefig(os.path.join(os.path.dirname(excel_path),savename + '_all_runs.png'))
    plt.show()

def doublingtime():
    ## calculates doubling time for each culture
    excel_paths, exp_name, OD_norm_data, use_fit, OD_exp_fit, OD_add_error_to_OD_plot, exp_names = importconfigOD()
    data_master = []
    if use_fit != True:
        OD_exp_fit = False
        adderrorbars = False
    for excel_path in excel_paths:
        print("\n\nData from"+ excel_path)
        data = import_data_OD(excel_path)
        data_master.append(data)
        print(data)
    if len(excel_paths) > 1:
        OD_norm_data = True
    data_cong, data_lengths, data_times, data_names, data_legends, data_no_cultures, data_no_perculture, data_no_timepoints, data_total_pos, name_legend_matches, data_master, data_names_legends = prepdata_data_multexp(data_master, OD_norm_data=False)

    #no_timepoints, total_pos, no_cultures, no_perculture = getmetadata_genstuff(data)
    #times = metadata_time(data, no_timepoints)
    #names = metadata_names(data, total_pos, no_perculture)
    #legend = metadata_legend(data, total_pos)
    #data = cut_data(data)
    #print(data)
    #print(exp_names)
    results_master = []
    for masterindex, data in enumerate(data_master):
        results = []
        no_perculture = data_no_perculture[masterindex]
        times = data_times[masterindex]
        names = data_names[masterindex]
        no_timepoints = data_no_timepoints[masterindex]
        legend = data_legends[masterindex]
        for i, culturename in enumerate(names):
            for j in range(i*no_perculture, (i+1)*no_perculture, 1):
                doubling_times = []
                for k in range(1, no_timepoints):
                    if data.iloc[j, k] > data.iloc[j, k-1]:
                        doubling_times.append((np.log(2) * (times[k] - times[k-1]))/ np.log(data.iloc[j, k] / data.iloc[j, k-1]))
                average = sum(doubling_times) / len(doubling_times)
                #print(culturename, legend[j], average)
                results.append([culturename, legend[j], average, data.iloc[j][0]])
        results = pd.DataFrame(results)
        results.rename(columns={results.columns[0]: 'Culture', results.columns[1]: 'Hormone conc.', results.columns[2]: 'Doubling time', results.columns[3]: 'Starting OD'}, inplace=True)
        saveexcel(results, os.path.join(os.path.dirname(excel_path), exp_names[masterindex]) + '_doublingtime_hardcalc.xlsx') 
        results_master.append(results)
    
    if use_fit == True:
        results_master_avg = results_master.copy()
        results_master = []
        if OD_norm_data == True:
            for i, datap in enumerate(data_master):
                data_master[i] = norm_data(datap, data_total_pos[i], data_no_timepoints[i])
        for i, matches in enumerate(name_legend_matches):
            OD_time = []
            for listindex, entryindex in matches:
                for k in range(data_no_timepoints[listindex]):
                    OD_time.append((data_master[listindex].iloc[entryindex][k], data_times[listindex][k]))
                #print(data_names_legends[listindex][entryindex])
                #printl(OD_time)
            listindex_root, entryindex_root = matches[0]
            name_entry, legend_entry = re.split(r"\$", data_names_legends[listindex_root][entryindex_root], 1)
            start_list = []
            pred_list = []
            for listindex, entryindex in matches:
                start_list.append(data_master[listindex].iloc[entryindex, 0])
                #printl(len(results_master_avg[listindex]))
                len(results_master_avg[listindex])
                for j in range(len(results_master_avg[listindex])):
                    data_part = data_names_legends[listindex][entryindex].replace("$", "\$").lower().lstrip().rstrip()
                    res_part = results_master_avg[listindex].iloc[j][0].lstrip().rstrip().lower() +"\$"+ results_master_avg[listindex].iloc[j][1].lower().lstrip().rstrip()
                    #print(listindex, entryindex, j)
                    #print(data_part, res_part)
                    if data_part == res_part:
                        printl("found smth")
                        print(listindex, entryindex)
                        print(data_part, res_part)
                        pred_list.append(results_master_avg[listindex].iloc[entryindex][2])
            start = sum(start_list)/len(start_list)
            pred = sum(pred_list)/len(pred_list)
            result = fitting_new([OD[0] for OD in OD_time], [time[1] for time in OD_time], start, pred, OD_exp_fit, name_entry, legend_entry)
            if OD_exp_fit == True:
                D = result.params[1]
                linoffset = result.params[0]
                D = (log(2)-linoffset)/D
                starting_OD_fit = exp(linoffset)*start
                partA = exp(linoffset)*start*linoffset
                starting_OD_fit_err = (result.bse[0]**2 * partA**2)**(1/2)
                partA = -1 / result.params[1]
                partB = -(np.log(2) - result.params[0]) / (result.params[1] ** 2)
                standard_error = sqrt((partA * result.bse[0])**2 + (partB * result.bse[1])**2)
            else:
                D = (2-result.params[0])/result.params[1]
                starting_OD_fit = result.params[0]*start
                partA = -1 / result.params[1]
                partB = -(2 - result.params[0]) / (result.params[1] ** 2)
                standard_error = sqrt((partA * result.bse[0])**2 + (partB * result.bse[1])**2)
                starting_OD_fit_err = result.bse[0]*start
            results_master.append([name_entry, legend_entry, D, starting_OD_fit, standard_error, starting_OD_fit_err])
        results_master = pd.DataFrame(results_master)
        print(results_master)
        #print(results_master.shape[1])
        results_master.rename(columns={results_master.columns[0]: 'Culture', results_master.columns[1]: 'Hormone conc.', results_master.columns[2]: 'Doubling time', results_master.columns[3]:"Starting culture size from fit", results_master.columns[4]: "Confidence intervals 95% coeff", results_master.columns[5]:"Conf. inv. lin. offset"}, inplace=True)
        saveexcel(results_master, os.path.join(os.path.dirname(excel_path), exp_name + '_doublingtime_fit.xlsx'))
    total_len = len(name_legend_matches)
    fig2, ax2 = plt.subplots(layout="constrained")
    #print(data_names)
    name_dict = find_cult_list(data_names)
    printl(name_dict, pretty=True)
    for i, culturename in enumerate(name_dict.values()[0]):
        coordintates = name_dict[culturename]
        colors = getcolormap()
        width = 1/(no_perculture+1)
        multiplier = 0.0
        index = np.arange(no_cultures)
        coordinates = []

    for i, culturename in enumerate(names):
        ###Plotting hormone conc vs doubling time
        k = i * no_perculture
        l = (i + 1) * no_perculture
        for j in range(k, l, 1):
            offset = width * multiplier   
            ax2.bar(offset, results.iloc[j][2], label=results.iloc[j][1], width=width)
            coordinates.append(offset)
            multiplier += 1
        multiplier += 1
        ###plotting fit
        fig, ax = plt.subplots()
        for j in range(i*no_perculture, (i+1)*no_perculture, 1):
            ax.scatter(times, data.iloc[j], marker='x', label=results.iloc[j][1], color=colors[j-i*no_perculture])
            y = []
            y_upper = []
            y_lower = []
            for time in times:
                if use_fit != True:
                    y.append(calc_OD(time, results.iloc[j][2], results.iloc[j][3]))
                elif OD_exp_fit == True:
                    y.append(2**(time/results.iloc[j][2]) * results.iloc[j][3])
                    if adderrorbars:
                        y_u, y_l = calcerrorslowerupper(calc_OD, time, (results.iloc[j][2], results.iloc[j][4]), (results.iloc[j][3], results.iloc[j][5]))
                        y_upper.append(y_u)
                        y_lower.append(y_l)
                        ###fit_curve(time, data.iloc[j, 0], log(2)/results.iloc[j][2]))
                elif OD_exp_fit != True:
                    #print(fit_curve_lin(time, data.iloc[j, 0], results.iloc[j][2]))
                    y.append(calc_OD_lin(time, results.iloc[j][2], results.iloc[j][3]))
                    if adderrorbars:
                        y_u, y_l = calcerrorslowerupper(calc_OD_lin, time, (results.iloc[j][2], results.iloc[j][4]), (results.iloc[j][3], results.iloc[j][5]))
                        y_upper.append(y_u)
                        y_lower.append(y_l)
                elif use_fit == True:
                    y.append(results.iloc[j][2]*time +  results.iloc[j][3])
            ax.plot(times, y, color=colors[j-i*no_perculture])
            if adderrorbars == True:
                ax.fill_between(times, y_upper, y_lower, color=colors[j-i*no_perculture], alpha=0.3)

        ax.set_title(culturename)
        ax.set_xlabel('Time (h)')
        if OD_norm_data == True:
            ax.set_ylabel('Normalized Optical Density')
        else:
            ax.set_ylabel('Optical Density')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
        if use_fit == True:
            fig.canvas.manager.set_window_title(exp_name + '_' + culturename + '_fit')
            #print(os.path.join(os.path.dirname(excel_path), exp_name + '_' + culturename))
            fig.savefig(os.path.join(os.path.dirname(excel_path), exp_name + '_' + culturename + '_fit.png'))
        else:
            fig.canvas.manager.set_window_title(exp_name + '_' + culturename+ '_basicfit')
            fig.savefig(os.path.join(os.path.dirname(excel_path), exp_name + '_' + culturename + '_basicfit.png'))

    
    ax2.set_title("Hormone concentration vs doubling-time")
    ax2.set_xticks(index + (((no_perculture-1)/2)* width) )
    ax2.set_xticklabels(names)
    ax2.set_xlabel('Culture')
    ax2.set_ylabel('Doubling time (h)')


    ax2 = labelreorg(ax2)
    if adderrorbars:
        ax2.errorbar(coordinates, results.iloc[:,2], yerr=results.iloc[:,4], capsize=4, color='black', ls="none")
    ax2.grid(True)
    fig2.canvas.manager.set_window_title(exp_name +  '_DoublingTimeHormConc')
    fig2.savefig(os.path.join(os.path.dirname(excel_path), exp_name + '_DoublingTimeHormConc.png'))
    plt.show()