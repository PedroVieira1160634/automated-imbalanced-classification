import PySimpleGUI as sg
from ml import execute_ml

#sg.theme('Dark Blue 3')

layout = [[sg.Text('Automated Imbalanced Classification')],
      [sg.Text('Dataset file location ', size=(15, 1)), sg.InputText(), sg.FileBrowse()],
      [sg.Submit(), sg.Cancel()]]

window = sg.Window('Automated Imbalanced Classification', layout)

while True:
    event, values = window.read()
    
    if event == sg.WIN_CLOSED or event == 'Cancel':
        break
    
    execute_ml(values[0])

window.close()
