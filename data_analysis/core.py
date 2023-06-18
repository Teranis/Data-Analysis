import os
import subprocess


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
