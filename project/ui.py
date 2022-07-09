import PySimpleGUI as sg
from ml import execute_byCharacteristics

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
            str_output = execute_byCharacteristics(values['file'], "")
        elif values['omid']:
            str_output = execute_byCharacteristics("", values['omid'])
        
        sg.Popup(str_output, keep_on_top=True, title='Recommendations')
    
    else:
        sg.Popup("Please choose a dataset file or put an OpenML dataset ID!\n", keep_on_top=True, title='Error')

window.close()
