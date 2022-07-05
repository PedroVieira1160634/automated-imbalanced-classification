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
    
    # print("event", event)
    # print("values", values)
    
    if event == sg.WIN_CLOSED or event == 'Cancel':
        break
    
    elif values['file'] or values['omid']:
        
        print("file\n", values['file'])
        print("omid\n", values['omid'])
        
        # values[0] to values['file']
        # or
        # values['omid']
        
        #validations
        
        # # print(values[0])
        # # sg.Popup('Ok clicked', keep_on_top=True)
        
        # # df_dist = execute_byCharacteristics(values[0], "")
        # # sg.Popup(df_dist, keep_on_top=True, title='Recommendations')
        
        # df_dist = pd.read_csv(sys.path[0] + "/input/" + "test_ui.csv", sep=",")
        
        # df_dist.loc[-1] = ['Pre Processing', 'Algorithm']
        # df_dist.index = df_dist.index + 1
        # df_dist = df_dist.sort_index()
        
        # str_output = "Top performing combinations of Pre Processing Technique with a Classifier Algorithm\n\n"
        # str_output += "\n".join("{:30} {:30}".format(x, y) for x, y in zip(df_dist['pre processing'], df_dist['algorithm']))
        # str_output += "\n"
        # sg.Popup(str_output, keep_on_top=True, title='Recommendations')
    
    else:
        sg.Popup("Please choose a Dataset File or an OpenML dataset ID!\n", keep_on_top=True, title='Error')

window.close()
