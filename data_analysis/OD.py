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
from core import  labelreorg, saveexcel, getcolormap, calcerrorslowerupper
from core import loadexcel as import_data_OD
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

#getmetadata_genstuff(data):
#    ## getting metadata
#    data.iloc[0, 0] = data.iloc[0, 0].replace(' ', '')

def fit_curve(time, start_OD, D):
    ## exp fit
    OD = start_OD*np.exp(time*D)
    return OD

def calc_OD(time, DT, startOD):
     return 2**(time/DT) * startOD

def calc_OD_lin(time, D, startOD):
     return D*time + startOD

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
        fitstartvals = [start_OD, fitstartval]
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

    print("\n\n" + culture_name + " " + legend)
    print(fitstartvals)
    print(results.summary())
    return results


### Main
def odplot():
    ## creates plots for each culture
    excel_path, exp_name, no_timepoints, no_perculture, no_cultures, total_pos, OD_norm_data, use_fit, OD_exp_fit, adderrorbars = importconfigOD()
    data = import_data_OD(excel_path)
    print(data)
    #no_cultures, no_perculture, no_timepoints, total_pos = getmetadata_genstuff(data)
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
    excel_path, exp_name, no_timepoints, no_perculture, no_cultures, total_pos, OD_norm_data, use_fit, OD_exp_fit, adderrorbars = importconfigOD()
    if use_fit != True:
        OD_exp_fit = False
        adderrorbars = False
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
            #print(culturename, legend[j], average)
            results.append([culturename, legend[j], average])
    results = pd.DataFrame(results)
    results.rename(columns={results.columns[0]: 'Culture', results.columns[1]: 'Hormone conc.', results.columns[2]: 'Doubling time'}, inplace=True)
    saveexcel(results, os.path.join(os.path.dirname(excel_path), exp_name) + '_doublingtime.xlsx') 
    if OD_norm_data == True:
        data = norm_data(data, total_pos, no_timepoints)

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
                ###Different implementations for the fit
                ###
                #print(type(results))
                #print(results)
                result = fitting_new(ODs, times, data.iloc[j, 0], results_avg.iloc[j][2], OD_exp_fit, culturename, legend[j])
                if OD_exp_fit == True:
                    D = result.params[1]
                    linoffset = result.params[0]
                    D = (log(2)-linoffset)/D
                    starting_OD_fit = exp(linoffset)*data.iloc[j, 0]
                    starting_OD_fit_err = (result.bse[0]**2 * result.params[0]**2)**(1/2)
                    partA = -1 / result.params[1]
                    partB = -(np.log(2) - result.params[0]) / (result.params[1] ** 2)
                    standard_error = sqrt((partA * result.bse[0])**2 + (partB *  result.bse[1])**2)
                else:
                    D = result.params[1]
                    standard_error = result.bse[1]
                    starting_OD_fit = result.params[0]
                    starting_OD_fit_err = result.bse[0]
                results.append([culturename, legend[j], D, starting_OD_fit, standard_error, starting_OD_fit_err])
        results = pd.DataFrame(results)
        results.rename(columns={results.columns[0]: 'Culture', results.columns[1]: 'Hormone conc.', results.columns[2]: 'Doubling time', results.columns[3]:"Starting culture size from fit", results.columns[4]: "Confidence intervals 95% coeff", results.columns[5]:"Conf. inv. lin. offset"}, inplace=True)
                ###
                
                    
                ###
                #if OD_exp_fit == True:
                #    D = fitting(fit_curve, ODs, np.array(times), data.iloc[j, 0], results_avg.iloc[j][2])
                #    D = log(2)/D
                #else:
                #    D = fitting(fit_curve_lin, ODs, np.array(times), data.iloc[j, 0], results_avg.iloc[j][2])
        #results.append([culturename, legend[j], D])
        #results = pd.DataFrame(results)
                ####

        saveexcel(results, os.path.join(os.path.dirname(excel_path), exp_name) + '_doublingtime_fit.xlsx')

    colors = getcolormap(no_perculture)
    width = 1/(no_perculture+1)
    multiplier = 0.0
    fig2, ax2 = plt.subplots(layout="constrained")
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
                    y.append(2**(time/results.iloc[j][2]) * data.iloc[j][0])
                elif OD_exp_fit == True:
                    y.append(2**(time/results.iloc[j][2]) * results.iloc[j][3])
                    if adderrorbars:
                        y_u, y_l = calcerrorslowerupper(calc_OD, time, (results.iloc[j, 2], results.iloc[j][4]), (results.iloc[j][3], results.iloc[j][5]))
                        y_upper.append(y_u)
                        y_lower.append(y_l)
                        ###fit_curve(time, data.iloc[j, 0], log(2)/results.iloc[j][2]))
                elif OD_exp_fit != True:
                    #print(fit_curve_lin(time, data.iloc[j, 0], results.iloc[j][2]))
                    y.append(fit_curve_lin(time, results.iloc[j][3], results.iloc[j][2]))
                    if adderrorbars:
                        y_upper, y_lower = calcerrorslowerupper(calc_OD_lin, time, (data.iloc[j, 0], results.iloc[j][2]), (results.iloc[j][4], results.iloc[j][5]))
            
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
        fig.canvas.manager.set_window_title(exp_name + '_' + culturename + '_fit')
        #print(os.path.join(os.path.dirname(excel_path), exp_name + '_' + culturename))
        fig.savefig(os.path.join(os.path.dirname(excel_path), exp_name + '_' + culturename + '_fit.png'))

    
    ax2.set_title("Hormone concentration vs doublingtime")
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