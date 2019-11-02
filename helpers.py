import numpy as np

def getKfromNPZ(npzFilePath):
    file = np.load(npzFilePath)
    return file[file.files[0]]
