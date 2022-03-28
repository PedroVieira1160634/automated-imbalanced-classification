import PySimpleGUI as sg
from ml import *

#sg.theme('Dark Blue 3')

layout = [[sg.Text('Automated Imbalanced Classification')],
      [sg.Text('Dataset file location ', size=(15, 1)), sg.InputText(), sg.FileBrowse()],
      [sg.Submit(), sg.Cancel()]]

window = sg.Window('Automated Imbalanced Classification', layout)

while True:
    event, values = window.read()
    
    if event == sg.WIN_CLOSED or event == 'Cancel':
        break
    
    dataset_name = values[0].split('/')[-1]
    df = read_file(values[0])
    x_train, x_test, y_train, y_test = train_test_split_func(df)
    classify_evaluate_write(x_train, x_test, y_train, y_test, dataset_name)

window.close()
