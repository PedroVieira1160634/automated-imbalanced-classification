import PySimpleGUI as sg
from ml import execute_byCharacteristics

#remove later
import sys
import pandas as pd

sg.theme('Dark Blue 3')

layout = [[sg.Text('Automated Imbalanced Classification')],
      [sg.Text('Dataset file location ', size=(15, 1)), sg.InputText(key="file"), sg.FileBrowse()],
      [sg.Text('OpenML dataset ID ', size=(15, 1)), sg.InputText(key="omid")],
      [sg.Submit(), sg.Cancel()]]

window = sg.Window('Automated Imbalanced Classification', layout)

while True:
    event, values = window.read()
    
    if event == sg.WIN_CLOSED or event == 'Cancel':
        break
    
    elif values['file'] and values['omid']:
        sg.Popup("Please only choose a dataset file or an OpenML dataset ID, not both!\n", keep_on_top=True, title='Error')
    
    elif values['file'] or values['omid']:
        
        if values['file']:
            # df_dist = execute_byCharacteristics(values['file'], "")
            print("file\n", values['file'])
        elif values['omid']:
            # df_dist = execute_byCharacteristics("", values['omid'])
            print("omid\n", values['omid'])
        
        # or test
        df_dist = pd.read_csv(sys.path[0] + "/input/" + "test_ui.csv", sep=",")
        
        
        df_dist.loc[-1] = ['Pre Processing', 'Algorithm']
        df_dist.index = df_dist.index + 1
        df_dist = df_dist.sort_index()
        
        #TODO
        # df_dist = df_dist.insert(loc=0, column='Rank', value=[1,2,3])
        
        str_output = "Top performing combinations of Pre Processing Technique with a Classifier Algorithm\n\n"
        str_output += "\n".join("{:30} {:30}".format(x, y) for x, y in zip(df_dist['pre processing'], df_dist['algorithm']))
        str_output += "\n"
        sg.Popup(str_output, keep_on_top=True, title='Recommendations')
    
    else:
        sg.Popup("Please choose a dataset file or put an OpenML dataset ID!\n", keep_on_top=True, title='Error')

window.close()
