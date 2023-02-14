import numpy as np
from sklearn.model_selection import train_test_split

#This program reads in the data from the "blink" and "no_blink" files and combines them into an array that can be processed by TensorFlow
import os
import numpy as np

def dataReader(path_blink, path_no_blink, percent_train):
    #Lists for storing the names of the .txt files
    txt_list_blink = []
    txt_list_no_blink = []
    numFiles = len(txt_list_blink + txt_list_no_blink)

    #Reads in the names of the files and stores them in their respecive list
    for iter in os.listdir(path_blink):
        if (iter.endswith(".txt")):
            txt_list_blink.append(path_blink+"/"+iter)
    for iter in os.listdir(path_no_blink):
        if (iter.endswith(".txt")):
            txt_list_no_blink.append(path_no_blink+"/"+iter)

    allFiles = txt_list_blink + txt_list_no_blink
    numFiles = len(txt_list_blink) + len(txt_list_no_blink)

    #Loops through each file looking for the max amount of rows 
    #We will use this to pad the end of all the smaller data entries with zeros so they are all the same dimmension
    max_length = 0
    numFilesUnused = 0
    for i in range(numFiles):
        currFile = np.loadtxt(allFiles[i], delimiter=",", skiprows=5, usecols=range(0))
        numRows = np.shape(currFile)
        temp = numRows[0]

        # if temp > 1000:
        #     numFilesUnused += 1
        #     continue

        #Simple method to keep track of largest file
        if temp > max_length:
            max_length = temp

    print(numFilesUnused)

    #allData will be what holds all of the EEG data
    #allData_Output will hold a correspending '1' for blinks and '0' for non blinks
    allData = np.empty(shape=(numFiles, max_length, 32))
    allData_Output = np.empty(shape=numFiles)

    #Now we can read in all the data and pad the smaller inputs with zeros
    for i in range(numFiles):

        #Adjust usecols to modify which channels will be read in
        currFile = np.loadtxt(allFiles[i], delimiter=",", skiprows=5, usecols=range(0,32))
        newData = np.array(currFile)
        padAmount = (max_length - (int(np.shape(newData)[0])))
        newData = np.pad(newData, ((0, padAmount),(0,0)), 'constant', constant_values=(0))
        allData[i] = newData
        print("%d Files Processed" % (i+1))

        #Store corresponding values for output
        if i < len(txt_list_blink):
            allData_Output[i] = 1
        else:
            allData_Output[i] = 0

    X_train, X_val, Y_train, Y_val = train_test_split(allData, allData_Output, train_size=percent_train, shuffle=True)
    return X_train, X_val, Y_train, Y_val

x1, x2, y1, y2 = dataReader("/Users/benjablonski/Desktop/DataReader/blink", "/Users/benjablonski/Desktop/DataReader/no_blink", .8, )

print(np.shape(x1))