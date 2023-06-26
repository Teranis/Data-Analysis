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
    CC_path = config['CC_path']
    CC_exp_name = config['CC_exp_name']
    culture_names = config['CC_culture_names']
    custom_order = config['CC_custom_order']
    CC_norm_data = config['CC_norm_data']
    CC_culm = config['CC_culm']
    CC_fit = config['CC_fit']
    return CC_path, CC_exp_name, culture_names, custom_order, CC_norm_data, CC_culm, CC_fit

def importconfigOD():
    parent_dir = os.path.dirname(__file__)
    config_file_path = os.path.join(parent_dir, 'configOD.json')
    with open(config_file_path, 'r') as input_file:
        config_raw = input_file.read()
    config_raw = config_raw.replace('\\','/')
    config = json.loads(config_raw)
    excel_folder_path = config['OD_excel_path']
    exp_name = config['OD_exp_name']
    OD_norm_data = config['OD_norm_data']
    use_fit = config['OD_use_fit']
    OD_exp_fit = config['OD_exp_fit']
    for file_name in os.listdir(excel_folder_path):
        if re.search(r"(=?(measure))(=?(.*\.xlsx)$)", file_name):
            excel_path = os.path.join(excel_folder_path, file_name)
    print(excel_path)
    OD_add_error_to_OD_plot = config['OD_add_error_to_OD_plot']
    return excel_path, exp_name, OD_norm_data, use_fit, OD_exp_fit, OD_add_error_to_OD_plot