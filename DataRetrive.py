import lunarossa as lr
import pandas as pd
import numpy as np
from numpy import array

channele_list = ['Datetime', 'SinkMin_AP', 'YawRate', 'Bs', 'Heel', 'Pitch', 'Lwy', 'Tws']
list_FFS = ['PortInFlapRam1_PressRam', 'PortOutFlapRam1_PressRam', 'StbdInFlapRam1_PressRam',
            'StbdOutFlapRam1_PressRam', 'PortInFlapFus_Ang', 'StbdInFlapFus_Ang']
list_autopilot = ['PortFlapAPActive', 'StbdFlapAPActive']
list_aero = ['StbdJibSheetRam1_Press', 'PortJibSheetRam1_Press', 'MainSheetRam_Press', 'MainSheetRam_Pos',
             'PortTravRam_Press', 'StbdTravRam_Press', 'ForestayPin_Load']
list_rudder = ['RudderRake_Ang', 'RudderRakeRam_PressB', 'RudderYaw_Ang']
list_FCS = ['FCS_StbdCant_Ang', 'FCS_PortCant_Ang']
channele_list.extend(list_FFS)
channele_list.extend(list_autopilot)
channele_list.extend(list_aero)
channele_list.extend(list_rudder)
channele_list.extend(list_FCS)


# channele_list = ['Datetime','Bs','PortInFlapRam1_PressRam','PortOutFlapRam1_PressRam','StbdInFlapRam1_PressRam','StbdOutFlapRam1_PressRam','RudderRakeAng','PortFlapAPActive','StbdFlapAPActive','StbdJibSheetRam1_Press','PortJibSheetRam1_Press','MainSheetRam_Press','RudderRakeRam_PressB','SinkMin_AP', 'RudderRake_Ang','PortInFlapFus_Ang','StbdInFlapFus_Ang','FCS_StbdCant_Ang','FCS_PortCant_Ang','MainSheetRam_Pos','Pitch','Heel','YawRate','PortTravRam_Press','StbdTravRam_Press']
# channele_list = ['Datetime','Bs','PortFlapAPActive','StbdFlapAPActive','StbdJibSheetRam1_Press','PortJibSheetRam1_Press','MainSheetRam_Press','PortOutFlapRam1_PressA','PortOutFlapRam1_PressB','PortInFlapRam1_PressA','PortInFlapRam1_PressB','StbdOutFlapRam1_PressA','StbdOutFlapRam1_PressB','StbdInFlapRam1_PressA','StbdInFlapRam1_PressB','RudderRakeRam_PressA','RudderRakeRam_PressB','SinkMin_AP', 'RudderRake_Ang', 'PortOutFlapFus_Ang', 'PortInFlapL_Ang','PortInFlapFus_Ang','StbdOutFlapFus_Ang','StbdInFlapFus_Ang','FCS_StbdCant_Ang','FCS_PortCant_Ang']
# channele_list = ['Datetime','Bs','FCS_StbdCant_Ang','FCS_PortCant_Ang']

def getDataSet():
    dataSet = loadData()
    return dataSet

def loadData():
    # GET DATA (return pandas dataframe)
    print("# 2021-03-10 #")
    data = lr.getAPIChannelValues('AC 75', '2021-03-10', channele_list, '50ms', '2021-03-10 14:30:00.00',
                                  '2021-03-10 18:10:00.00')
    df_merged = cleanup(data, 0)
    print("# 2021-03-12 #")
    data = lr.getAPIChannelValues('AC 75', '2021-03-12', channele_list, '50ms', '2021-03-12 14:30:00.00',
                                  '2021-03-12 18:10:00.00')
    data = cleanup(data, 1)
    df_merged = pd.concat([df_merged, data], ignore_index=True, sort=False)
    print("# 2021-03-13 #")
    data = lr.getAPIChannelValues('AC 75', '2021-03-13', channele_list, '50ms', '2021-03-13 14:30:00.00',
                                  '2021-03-13 18:10:00.00')
    data = cleanup(data, 2)
    df_merged = pd.concat([df_merged, data], ignore_index=True, sort=False)
    print("# 2021-03-14 #")
    data = lr.getAPIChannelValues('AC 75', '2021-03-14', channele_list, '50ms', '2021-03-14 14:30:00.00',
                                  '2021-03-14 18:10:00.00')
    data = cleanup(data, 3)
    df_merged = pd.concat([df_merged, data], ignore_index=True, sort=False)
    print("# 2021-03-15 #")
    data = lr.getAPIChannelValues('AC 75', '2021-03-15', channele_list, '50ms', '2021-03-15 14:30:00.00',
                                  '2021-03-15 18:10:00.00')
    data = cleanup(data, 4)
    df_merged = pd.concat([df_merged, data], ignore_index=True, sort=False)
    print("# 2021-03-16 #")
    data = lr.getAPIChannelValues('AC 75', '2021-03-16', channele_list, '50ms', '2021-03-16 14:30:00.00',
                                  '2021-03-16 18:10:00.00')
    data = cleanup(data, 5)
    df_merged = pd.concat([df_merged, data], ignore_index=True, sort=False)
    print("# 2021-03-17 #")
    data = lr.getAPIChannelValues('AC 75', '2021-03-17', channele_list, '50ms', '2021-03-17 14:30:00.00',
                                  '2021-03-17 18:10:00.00')
    data = cleanup(data, 6)
    df_merged = pd.concat([df_merged, data], ignore_index=True, sort=False)
    print("# DONE Load from Server #")

    df_merged['isPort'] = df_merged.apply(isPortTack, axis=1)
    df_merged['isStbd'] = df_merged.apply(isStbdTack, axis=1)
    df_merged['armDown'] = df_merged.apply(isBothDown, axis=1)

    # df_merged = setFCS(df_merged,'FCS_StbdCant_Ang')
    # df_merged = setFCS(df_merged, 'FCS_PortCant_Ang')

    return df_merged


def cleanup(df, index):
    n = 100 # 5 seconds
    dataframe = df.copy()
    dataframe['isTowing'] = dataframe.apply(isTowing, axis=1)
    dataframe['isFloating'] = dataframe.apply(isFloating, axis=1)
    dataframe['isSailing'] = dataframe.apply(isSailing, axis=1)

    dataframe.drop(dataframe.loc[dataframe['isTowing'] == 1].index, inplace=True)
    dataframe.drop(dataframe.loc[dataframe['isFloating'] == 1].index, inplace=True)
    dataframe.drop(dataframe.loc[dataframe['isSailing'] == 0].index, inplace=True)
    dataframe = dataframe.drop(['isTowing', 'isFloating', 'isSailing', 'PortFlapAPActive', 'StbdFlapAPActive'], axis=1)

    dataframe.dropna(inplace=True)

    # Using drop() function to delete first n rows
    dataframe.drop(dataframe.tail(n).index, inplace=True)  # drop last n rows
    dataframe.drop(dataframe.head(n).index, inplace=True)  # drop first n rows

    avg = (int)(dataframe['Tws'].mean())
    dataframe['Tws'] = avg
    dataframe['Day'] = index

    dataframe.dropna(inplace=True)
    return dataframe


def manualImput(variable, df, windows):
    X = list()
    for i in range(len(df)):
        # find the end of this pattern
        end_ix = i + windows
        # check if we are beyond the dataset
        if end_ix > len(df):
            break
        # gather input and output parts of the pattern
        manualValue = df[i:end_ix, variable]
        seq_x = df[i:end_ix, :-1]
        X.append(seq_x)
    return array(X)


def isSailing(df):
    if (df['isTowing'] == 1):
        return 0
    elif (df['isFloating'] == 1):
        return 0
    elif (df['Bs'] <=18):
        return 0
    else:
        return 1


def isFloating(df):
    if (df['isTowing'] == 1):
        return 0
    elif ((df['FCS_StbdCant_Ang'] < 20) & (df['FCS_PortCant_Ang'] < 20) & (df['MainSheetRam_Press'] < 50) & (
            df['StbdJibSheetRam1_Press'] < 50) & (df['PortJibSheetRam1_Press'] < 50)):
        return 1
    else:
        return 0


def isTowing(df):
    if ((df['PortFlapAPActive'] > 0) & (df['StbdFlapAPActive'] > 0)):
        return 1
    else:
        return 0


def isPortTack(df):
    if (df['FCS_StbdCant_Ang'] < 71):
        return 1
    else:
        return 0


def isStbdTack(df):
    if (df['FCS_PortCant_Ang'] < 71):
        return 1
    else:
        return 0


def isBothDown(df):
    if (df['isPort'] == 1 and df['isStbd'] == 1):
        return 1
    else:
        return 0


def setFCS(df, label):
    conditionsStbd = [
        (df[label] < 5),
        (df[label] >= 5) & (df[label] < 50),
        (df[label] >= 50)
    ]
    values = [0, 1, 2]
    df[label] = np.select(conditionsStbd, values)
    return df

FILE_NAME = "full_file_sailing_noman.pkl" #full_file_sailing.pkl

def saveDataSet(dataSet):
    dataSet.to_pickle(FILE_NAME)
    print("# DONE DataSet Saved #")


def retriveDataSet(reload):
    try:
        if (reload != True):
            dataSet = pd.read_pickle(FILE_NAME)
        else:
            dataSet = None
    except:
        dataSet = None

    print("# DONE Load from File #")
    return dataSet


###############
# Some Examples #
#################

"""
#GET LIST OF AVAILABLE BOATS
boats = lr.getAPIBoats()
#print(boats)

#GET LIST OF AVAILABLE DATES BY BOAT
dates = lr.getAPIDates('AC 75')
print(dates)

#GET LIST OF SESSIONS BY DATE AND BOAT
sessions = lr.getAPISessions('2021-03-16' , 'AC 75')
print(sessions)


#GET LIST OF CHANNELS BY SESSION
name = str(sessions[1]['session']['name'])
url = str(sessions[1]['session']['url'])
source = 'ECC0'
channels = lr.getAPIChannels(name,url,source)
print(channels)
name = str(sessions[1]['session']['name'])
url = str(sessions[1]['session']['url'])
source = 'ECC1'
channels = lr.getAPIChannels(name,url,source)
print(channels)
"""
