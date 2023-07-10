import numpy as np
import pandas as pd
import os
from configload import importconfigspotMAX
import regex as re
from core import loadcsv, printl

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
###Main
def boxplot():
    config = importconfigspotMAX()
    paths = config["path"]
    for path in paths:
        folder_filter = config["folder_filter"]
        spotMAX_foldername = config["spotMAX_foldername"]
        spotMAX_filename = config["spotMAX_filename"]
        base_xlsx_files_paths, xlsx_files_paths = finding_xlsx(path, folder_filter, spotMAX_foldername, spotMAX_filename)
        printl(base_xlsx_files_paths, xlsx_files_paths)
        for file_path in base_xlsx_files_paths:
            printl(file_path, pretty=True)
            data = loadcsv(file_path)
            print(data)