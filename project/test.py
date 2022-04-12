print("inicio")

class Res(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

resultsList = []
for x in range(5):
    r1 = Res(x)
    resultsList.append(r1)

print(resultsList[0].dataset_name)

best_result=None
for result in resultsList:
    if(result.dataset_name == 1):
        best_result = result

print(best_result.dataset_name)

print("fim")