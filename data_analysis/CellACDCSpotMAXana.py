import pandas as pd
import numpy as np
import os
import regex as re
import warnings
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
funcval = 1
if funcval == 1:
    spotMAX_filename = "0_1_valid_spots_aggregated.csv"
    savepath = r"G:\Timon\Snapshots\5_Mip1_inducible_test_0\2023-07-25_FPY010_test_1"
    path_master = [r"G:\Timon\Snapshots\5_Mip1_inducible_test_0\2023-07-25_FPY010_test_1\FPY010-6_0nM", r"G:\Timon\Snapshots\5_Mip1_inducible_test_0\2023-07-25_FPY010_test_1\FPY010-6_20nM", r"G:\Timon\Snapshots\5_Mip1_inducible_test_0\2023-07-25_FPY010_test_1\FPY010-8_20nM"]
else:
    spotMAX_filename = "2_0_detected_spots_aggregated.csv"
    savepath = r"C:\Users\SchmollerLab\Documents\Timon\timelapse_tinca_completed"
    path_master = [r"C:\Users\SchmollerLab\Documents\Timon\timelapse_tinca_completed\2022-08-04", r"C:\Users\SchmollerLab\Documents\Timon\timelapse_tinca_completed\2022-10-20", r"C:\Users\SchmollerLab\Documents\Timon\timelapse_tinca_completed\2022-08-09"]

title1 = r"Mitochondrial network size"
title2 = r"Cell size"
title3 = r"Concentration of mitochondrial network"
title4 = r"Spots per cell"
title5 = r"Mitochondrial network per spot"

ylabel1 = r"Volume (p.d.u.)"
ylabel2 = r"Volume (fL)"
ylabel3 = "Concentration of \nmitochondiral network (p.d.u./fL)"
ylabel4 = r"Number of mtDNA nucleoid"
ylabel5 = "mtDNA nucleoides per volume of \nmitochondiral network (1/p.d.u.)"

folder_filter =  "Pos"
spotMAX_foldername = "spotMAX_output"


names = ['FPY010-6_0nM', 'FPY010-6_20nM', 'FPY010-8_20nM']
def finding_xlsx(paths, folder_filter, spotMAX_foldername, spotMAX_filename):
    #finds xlsx if structure is right
    base_xlsx_files_paths =[]
    xlsx_files_paths = []
    nametuples = []
    for path in paths:
        basename = os.path.basename(path)
        folder_list = os.listdir(path)
        
        folderfile_list = [(os.path.join(path, folder_name, spotMAX_foldername), folder_name) for folder_name in folder_list if folder_name.lower().startswith(folder_filter.lower())]
        #import pdb; pdb.set_trace()
        for folder_name, pos_name in folderfile_list:
            try:
                folder_cont = os.listdir(folder_name)
            except:
                print("\nDid not find a spotMAX analysis for:\n" + folder_name)
                continue
            for file_name in folder_cont:
                if re.match(spotMAX_filename, file_name):
                    base_xlsx_files_paths.append(os.path.join(folder_name, file_name))
                    xlsx_files_paths.append(folder_name)
                    nametuples.append((basename, pos_name))
    return base_xlsx_files_paths, xlsx_files_paths, nametuples

def addcolorbar(ax, sm, boundaries, legstart):
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_ticks(boundaries[:-1] + 0.5)
    cbar.set_ticklabels(boundaries[:-1])  # Set custom tick labels for the bins
    cbar.set_label("Generational difference from switch", fontsize=13) 
    legend2 = ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()
    custom_legend = plt.legend(handles[legstart:], labels[legstart:], title="")
    ax.add_artist(custom_legend)
    return ax


def pltstuff(final_df, before_df, title1, title2, title3, savepathext, part=False):
    #creates the plots <3 sns
    #also does frame time conv
    final_df = final_df.copy()
    final_df["division_frame_i"] = final_df["division_frame_i"]*(1/6) - 3
    if part == True:
        savepath = os.path.join(savepathext, 'single')
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            print('Created dir: ' + savepath)
    else:
        savepath = savepathext
    #Plotting mit. network (ref_ch_vol_um3)

    # Define the boundaries to create discrete bins
    boundaries = np.arange(final_df['gen_num_diff_from_switch'].min(), final_df['gen_num_diff_from_switch'].max() + 2) ##WHY HERE ?!?!ß
    legstart = final_df['gen_num_diff_from_switch'].max() + 3 ##WHY HERE 3?!?!

    # Extract colors from the "viridis" colormap
    n_colors = len(boundaries) - 1
    viridis_colors = plt.cm.viridis(np.linspace(0, 1, n_colors))

    # Create a custom colormap with discrete colors based on "viridis"
    cmap = mcolors.ListedColormap(viridis_colors)

    # Create a BoundaryNorm to map values to colors
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) 

    fig1, ax1 = plt.subplots()
    fig12, ax12 = plt.subplots()
    #import pdb; pdb.set_trace()
    #sns.scatterplot(data=before_df, x='division_frame_i', y='ref_ch_vol_um3', color=viridis_colors[0], marker='o', ax=ax12)
    sns.scatterplot(data=final_df, x='division_frame_i', y='ref_ch_vol_um3', hue='gen_num_diff_from_switch', style='hue', ax=ax12,  palette="viridis")
    sns.boxplot(data=final_df, x='gen_num_diff_from_switch', y='ref_ch_vol_um3', hue='hue', ax=ax1)
    ax12.set_ylabel(ylabel1, fontsize=13)
    ax1.set_ylabel(ylabel1, fontsize=13)
    ax12.set_xlabel('Time after media switch (h)', fontsize=13)
    ax1.set_xlabel('Relative generation following media change', fontsize=13)
    ax12 = addcolorbar(ax12, sm, boundaries, legstart)
    legend = ax1.legend()
    legend.set_title('')
    ax1.grid(True)
    ax12.grid(True)
    ax1.set_title(title1, fontsize=14, fontweight='bold')
    ax12.set_title(title1, fontsize=14, fontweight='bold')
    fig1.canvas.manager.set_window_title(title1)
    fig12.canvas.manager.set_window_title(title1)
    save_path1 = os.path.join(savepath, title1.replace(' ', '_') + '_boxplot.pdf')
    save_path2 = os.path.join(savepath, title1.replace(' ', '_') + '_scatter.pdf')
    fig1.savefig(save_path1)
    fig12.savefig(save_path2)
    print('Saved box-plot to ' + save_path1)
    print('Saved scatter to ' + save_path2)

    #Plotting mit. network (cell_vol_fl)
    fig2, ax2 = plt.subplots()
    fig22, ax22 = plt.subplots()
    sns.scatterplot(data=final_df, x='division_frame_i', y='cell_vol_fl', hue='gen_num_diff_from_switch', style='hue', ax=ax22, palette="viridis")
    sns.boxplot(data=final_df, x='gen_num_diff_from_switch', y='cell_vol_fl', hue='hue', ax=ax2)
    ax2.set_ylabel(ylabel2, fontsize=13)
    ax2.set_xlabel('Relative generation following media change', fontsize=13)
    ax22.set_ylabel(ylabel2, fontsize=13)
    ax22.set_xlabel('Time after media switch (h)', fontsize=13)
    legend = ax2.legend()
    legend.set_title('')
    ax22 = addcolorbar(ax22, sm, boundaries, legstart)
    ax2.grid(True)
    ax2.set_title(title2, fontsize=14, fontweight='bold')
    ax22.grid(True)
    ax22.set_title(title2, fontsize=14, fontweight='bold')
    fig2.canvas.manager.set_window_title(title2)
    save_path1 = os.path.join(savepath, title2.replace(' ', '_') + '_boxplot.pdf')
    fig2.savefig(save_path1)
    print('Saved box-plot to ' + save_path1)
    fig22.canvas.manager.set_window_title(title2)
    save_path2 = os.path.join(savepath, title2.replace(' ', '_') + '_scatter.pdf')
    fig22.savefig(save_path2)
    print('Saved scatter to ' + save_path2)

    #Plotting mit. network conc (mito_concentration)
    fig3, ax3 = plt.subplots()
    fig32, ax32 = plt.subplots()
    sns.scatterplot(data=final_df, x='division_frame_i', y='mito_concentration', hue='gen_num_diff_from_switch', style='hue', ax=ax32, palette="viridis")
    sns.boxplot(data=final_df, x='gen_num_diff_from_switch', y='mito_concentration', hue='hue', ax=ax3)
    ax3.set_ylabel(ylabel3, fontsize=13)
    ax3.set_xlabel('Relative generation following media change', fontsize=13)
    ax32.set_ylabel(ylabel3, fontsize=13)
    ax32.set_xlabel('Time after media switch (h)', fontsize=13)
    legend = ax3.legend()
    legend.set_title('')
    ax32 = addcolorbar(ax32, sm, boundaries, legstart)
    ax3.grid(True)
    ax3.set_title(title3, fontsize=14, fontweight='bold')
    ax32.grid(True)
    ax32.set_title(title3, fontsize=14, fontweight='bold')
    fig3.canvas.manager.set_window_title(title3)
    save_path1 = os.path.join(savepath, title3.replace(' ', '_')  + '_boxplot.pdf')
    fig3.savefig(save_path1)
    print('Saved box-plot to ' + save_path1)
    fig32.canvas.manager.set_window_title(title3)
    save_path2 = os.path.join(savepath, title3.replace(' ', '_') + '_scatter.pdf')
    fig32.savefig(save_path2)
    print('Saved scatter to ' + save_path2)

plt.rc('axes', axisbelow=True)
if funcval == 0:
    ###main
    base_xlsx_files_paths, xlsx_files_paths, nametuples = finding_xlsx(path_master, folder_filter, spotMAX_foldername, spotMAX_filename)
    final_df_master = pd.DataFrame()
    final_df_before_master = pd.DataFrame() 
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        print('Created dir: ' + savepath) 
    for indx, xlsx_files_path in enumerate(base_xlsx_files_paths):
        print("\nPlotting: "+ xlsx_files_path)
        df = pd.read_csv(xlsx_files_path)
        df['mito_concentration'] = df.ref_ch_vol_um3/df.cell_vol_fl #Calculation for rel conc.
        # Get relevant columns
        columns = [
            'frame_i', 
            'Cell_ID', 
            'cell_vol_fl',
            'ref_ch_vol_um3',
            'cell_cycle_stage',
            'relative_ID', 
            'generation_num',
            'relationship',
            'division_frame_i',
            'mito_concentration'
        ]
        df = df[columns]

        # Get annotated part of the table
        df = df.loc[df.cell_cycle_stage.dropna().index]

        # import pdb; pdb.set_trace()

        # Get cells after medium switch
        df_after_switch = df[df.frame_i >= 18]
        df_before_switch = df[df.frame_i < 18]
        # Get mother cells at switch
        df_mother_switch = df_after_switch[
            (df_after_switch.frame_i == 18) & (df_after_switch.relationship == 'mother')
        ]

        # Get daughter cells after switch (first G1 frame)
        df_after_switch_daughter = df_after_switch[
            (df_after_switch.generation_num == 1) & (df_after_switch.cell_cycle_stage == 'G1') & (df_after_switch.division_frame_i >= 18)
        ]
        df_after_switch_daughter_at_birth = (
            df_after_switch_daughter
            .sort_values(['Cell_ID', 'frame_i'])
            .groupby('Cell_ID')
            .first()
            .reset_index()
        )

        # Get mother cells after switch (first G1 frame)
        df_after_switch_mother = df_after_switch[
            (df_after_switch.generation_num > 1) 
            & (df.cell_cycle_stage == 'G1') 
            & (df.division_frame_i >= 18)
        ]

        df_after_switch_mother = (
            df_after_switch_mother
            .set_index(['Cell_ID', 'frame_i'])
        )

        # Get generation number of the mothers 
        df_after_switch_daughter_at_birth = df_after_switch_daughter_at_birth.set_index(['relative_ID', 'frame_i'])


        gen_number_of_mothers = (
            df_after_switch.set_index(['Cell_ID', 'frame_i'])
            .reindex(df_after_switch_daughter_at_birth.index)
            .generation_num
        )


        # Add generation number of mothers column to df_after_switch_daughter_cells
        df_after_switch_daughter_at_birth.loc[gen_number_of_mothers.index, 'generation_num_mother'] = gen_number_of_mothers
        df_after_switch_daughter_at_birth = df_after_switch_daughter_at_birth.dropna()
        df_after_switch_daughter_at_birth['generation_num_mother'] = df_after_switch_daughter_at_birth['generation_num_mother'].astype(int)

        df_after_switch_daughter_at_birth = df_after_switch_daughter_at_birth.reset_index().set_index(['relative_ID', 'generation_num_mother'])

        keys = []
        dfs = []
        # Iterate next generations of mother cells at switch
        for i in range(5):
            df_mother_switch['next_gen_after_switch'] = df_mother_switch['generation_num'] + i
            df_mother_switch_indexed = df_mother_switch.set_index(['Cell_ID', 'next_gen_after_switch'])
            
            df_daughter_of_mother_at_switch = df_after_switch_daughter_at_birth.filter(df_mother_switch_indexed.index, axis=0)

            index_mother = list(zip([sublist[0] for sublist in df_daughter_of_mother_at_switch.index], df_daughter_of_mother_at_switch['frame_i']))
            df_after_switch_mother_indx = df_after_switch_mother.filter(index_mother, axis=0)

            keys.append((i, 'mother'))
            keys.append((i, 'daughter'))

            dfs.append(df_after_switch_mother_indx .reset_index())
            dfs.append(df_daughter_of_mother_at_switch.reset_index())
            
            #import pdb; pdb.set_trace()
        final_df = pd.concat
        final_df = pd.concat(dfs, keys=keys, names=['gen_num_diff_from_switch', 'hue']).reset_index()

        #changing the titles to be unique


        title1_loc = nametuples[indx][0] + "_" + nametuples[indx][1].rstrip('.csv') + "_" + title1
        title2_loc = nametuples[indx][0] + "_" + nametuples[indx][1].rstrip('.csv') + "_" + title2
        title3_loc = nametuples[indx][0] + "_" + nametuples[indx][1].rstrip('.csv') + "_" + title3

        #pltstuff(final_df, df_before_switch, title1_loc, title2_loc, title3_loc, savepath, part=True)

        final_df_master = pd.concat([final_df_master, final_df])
        final_df_before_master = pd.concat([final_df_before_master, df_before_switch])
    print(df.groupby(['hue', 'generation']).describe())
    print("\nPlotting all together!:")
    pltstuff(final_df_master, final_df_before_master, title1, title2, title3, savepath)

    plt.show()

if funcval == 1:
    ###main
    base_xlsx_files_paths, xlsx_files_paths, nametuples = finding_xlsx(path_master, folder_filter, spotMAX_foldername, spotMAX_filename)
    final_df_master = pd.DataFrame()
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        print('Created dir: ' + savepath) 
    for indx, xlsx_files_path in enumerate(base_xlsx_files_paths):
        print("\nLoading: "+ xlsx_files_path)
        df = pd.read_csv(xlsx_files_path)
        df['mito_concentration'] = df.ref_ch_vol_um3/df.cell_vol_fl #Calculating mito per cell vol
        df['spots_per_mito'] = df.num_spots/df.ref_ch_vol_um3 #Calculating mito vol per spot
        # Get relevant columns
        columns = [
            'cell_vol_fl',
            'ref_ch_vol_um3',
            'mito_concentration',
            'num_spots',
            'spots_per_mito'
        ]
        df = df[columns]
        file_name = os.path.abspath(xlsx_files_path)
        
        file_components = file_name.split('\\')
        desired_part = file_components[-4].replace('_', ' ')
        
        df['name'] = desired_part
        final_df_master = pd.concat([final_df_master, df])


    final_df_master['hue'] =  final_df_master['name'].str.split(' ').str[-1]


    fig, ax = plt.subplots()
    sns.boxplot(data=final_df_master, y='ref_ch_vol_um3', x='name', ax=ax, hue='hue', dodge=False)
    ax.set_ylabel(ylabel1, fontsize=13)
    ax.set_xlabel('Culture name', fontsize=13)
    ax.grid(True)
    ax.set_title(title1, fontsize=14, fontweight='bold')
    ax.legend().remove()
    fig.canvas.manager.set_window_title(title1)
    save_path1 = os.path.join(savepath, title1.replace(' ', '_')  + '_exp2.pdf')
    fig.savefig(save_path1)
    print('Saved box-plot to ' + save_path1)

    fig, ax = plt.subplots()
    sns.boxplot(data=final_df_master, y='cell_vol_fl', x='name', ax=ax, hue='hue', dodge=False)
    ax.set_ylabel(ylabel2, fontsize=13)
    ax.set_xlabel('Culture name', fontsize=13)
    ax.grid(True)
    ax.set_title(title2, fontsize=14, fontweight='bold')
    ax.legend().remove()
    fig.canvas.manager.set_window_title(title2)
    save_path1 = os.path.join(savepath, title2.replace(' ', '_')  + '_exp2.pdf')
    fig.savefig(save_path1)
    print('Saved box-plot to ' + save_path1)

    fig, ax = plt.subplots()
    sns.boxplot(data=final_df_master, y='mito_concentration', x='name', ax=ax, hue='hue', dodge=False)
    ax.set_ylabel(ylabel3, fontsize=13)
    ax.set_xlabel('Culture name', fontsize=13)
    ax.grid(True)
    ax.set_title(title3, fontsize=14, fontweight='bold')
    ax.legend().remove()
    fig.canvas.manager.set_window_title(title3)
    save_path1 = os.path.join(savepath, title3.replace(' ', '_')  + '_exp2.pdf')
    fig.savefig(save_path1)
    print('Saved box-plot to ' + save_path1)

    fig, ax = plt.subplots()
    sns.boxplot(data=final_df_master, y='num_spots', x='name', ax=ax, hue='hue', dodge=False)
    ax.set_ylabel(ylabel4, fontsize=13)
    ax.set_xlabel('Culture name', fontsize=13)
    ax.grid(True)
    ax.set_title(title4, fontsize=14, fontweight='bold')
    ax.legend().remove()
    fig.canvas.manager.set_window_title(title4)
    save_path1 = os.path.join(savepath, title4.replace(' ', '_')  + '_exp2.pdf')
    fig.savefig(save_path1)
    print('Saved box-plot to ' + save_path1)

    fig, ax = plt.subplots()
    sns.boxplot(data=final_df_master, y='spots_per_mito', x='name', ax=ax, hue='hue', dodge=False)
    ax.set_ylabel(ylabel5, fontsize=13)
    ax.set_xlabel('Culture name', fontsize=13)
    ax.grid(True)
    ax.set_title(title5, fontsize=14, fontweight='bold')
    ax.legend().remove()
    fig.canvas.manager.set_window_title(title5)
    save_path1 = os.path.join(savepath, title5.replace(' ', '_')  + '_exp2.pdf')
    fig.savefig(save_path1)
    print('Saved box-plot to ' + save_path1)

    summary = final_df_master.describe()
    savepathexl = os.path.join(savepath, 'results.xlsx')
    summary.to_excel(savepathexl)
    print(f"Summary statistics saved to '{savepathexl}'")
    plt.show()