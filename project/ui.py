import PySimpleGUI as sg
from ml import execute_byCharacteristics

sg.theme('Dark Blue 3')

layout = [[sg.Text('Automated Imbalanced Classification')],
      [sg.Text('Dataset file location ', size=(15, 1)), sg.InputText(), sg.FileBrowse()],
      [sg.Submit(), sg.Cancel()],
      [sg.Text("", size=(0, 1), key='OUTPUT')]]

window = sg.Window('Automated Imbalanced Classification', layout)

while True:
    event, values = window.read()
    
    if event == sg.WIN_CLOSED or event == 'Cancel':
        break
    
    print(values[0])
    window['OUTPUT'].update(value=values[0])
    
    # df_dist = execute_byCharacteristics(values[0], "")
    # window['OUTPUT'].update(value=df_dist)

window.close()
