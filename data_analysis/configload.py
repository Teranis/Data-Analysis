import os
import json
import re


def importconfigCC():
    parent_dir = os.path.dirname(__file__)
    config_file_path = os.path.join(parent_dir, 'configCC.json')
    with open(config_file_path, 'r') as input_file:
        config_raw = input_file.read()
    config_raw = config_raw.replace('\\','/')
    config = json.loads(config_raw)
    CC_paths = config['CC_path']
    CC_exp_names = config['CC_exp_name']
    culture_names = config['CC_culture_names']
    custom_order = config['CC_custom_order']
    CC_norm_data = config['CC_norm_data']
    CC_culm = config['CC_culm']
    CC_fit = config['CC_fit']
    exp_name_master = config['exp_name_master']
    savepath = config['save_path']
    if savepath is not str or savepath=="":
        savepath = CC_paths[0]
    if exp_name_master is not str or exp_name_master=="":
        exp_name_master = CC_exp_names[0]
    return CC_paths, CC_exp_names, culture_names, custom_order, CC_norm_data, CC_culm, CC_fit, savepath, exp_name_master

def importconfigOD():
    parent_dir = os.path.dirname(__file__)
    config_file_path = os.path.join(parent_dir, 'configOD.json')
    with open(config_file_path, 'r') as input_file:
        config_raw = input_file.read()
    config_raw = config_raw.replace('\\','/')
    config = json.loads(config_raw)
    excel_folder_paths = config['OD_excel_path']
    exp_name = config['OD_exp_name']
    OD_norm_data = config['OD_norm_data']
    use_fit = config['OD_use_fit']
    OD_exp_fit = config['OD_exp_fit']
    excel_paths = []
    exp_names = []
    for excel_folder_path in excel_folder_paths:
        for file_name in os.listdir(excel_folder_path):
            if re.search(r"(=?(measure))(=?(.*\.xlsx)$)", file_name):
                excel_paths.append(os.path.join(excel_folder_path, file_name))
                print(file_name)
                exp_names.append(re.findall(r"\d{1,2}_\d{1,2}_\d{1,2}_?\d?", file_name)[0])
    OD_add_error_to_OD_plot = config['OD_add_error_to_OD_plot']
    return excel_paths, exp_name, OD_norm_data, use_fit, OD_exp_fit, OD_add_error_to_OD_plot, exp_names