from datetime import datetime as dt, timedelta
import requests
import pandas as pd
import numpy as np
import json
import os
import math as m


# import LRbrain.Statistic as st
# from websocket import create_connection

def left(s, amount):
    return s[:amount]


def right(s, amount):
    return s[-amount:]


def mid(s, offset, amount):
    return s[offset:offset + amount]


def stripDate(date_in):
    return date_in.replace("-", "")


def getDate(date_in):
    ti = date_in.find("T")
    return left(date_in, ti)


def cleanTime(time_in):
    time_str = time_in.replace("T", " ").replace("Z", "")
    return time_str


def clense(val):
    try:
        if type(val) == str:
            no_dot = val.replace('.', '')
            if no_dot.isnumeric():
                num = float(val)

                return num
            else:
                return 0
        else:
            return val
    except:
        return 0


def getAav(data):
    try:
        s = 0
        pd = 0
        i = 0
        for d in data:
            if (i == 0):
                pd = d

            s += abs(d - pd)
            pd = d
            i += 1
        return float(s / (i / 50))
    except:
        return 0


def AngleBetween(first, second):
    try:
        first = float(first)
        second = float(second)

        if (first >= 0 or second >= 0 or first <= 0 or second <= 0):
            if (first > 720 or second > 720 or first < -720 or second < -720):
                return 0

            between = first - second

            while between > 180:
                between -= 360

            while between < -180:
                between += 360

            return between
        else:
            if (first < 0):
                first = abs(first) + 180
            else:
                second = abs(second) + 180

            if (first > 720 or second > 720 or first < -720 or second < -720):
                return 0

            between = first - second

            while between > 180:
                between -= 360

            while between < -180:
                between += 360

            return between
    except:
        return 0


def AngleSubtract(first, second):
    try:
        first = float(first)
        second = float(second)

        if (first > 720 or second > 720 or first < -720 or second < -720):
            return 0

        subtract = first - second

        while subtract > 360:
            subtract -= 360

        while subtract < -360:
            subtract += 360

        return subtract
    except:
        return 0


def AngleNormalize(angle):
    try:
        if (angle > 360):
            return angle - 360
        elif (angle < 0):
            return angle + 360

        return angle
    except:
        return 0


def LatLontoMeters(lat, lon):
    originlat = 39.120657
    originlon = 9.181797

    m_per_deg_lat = 111132.954 - 559.822 * m.cos(2 * m.radians(originlat)) + 1.175 * m.cos(4 * m.radians(originlat))
    m_per_deg_lon = 111132.954 * m.cos(m.radians(originlat))

    x = (originlon - lon) * m_per_deg_lon
    y = (originlat - lat) * m_per_deg_lat

    return [x, y]


def MeterstoLatLon(x, y):
    originlat = 39.120657
    originlon = 9.181797

    deltaLat = m.degrees(m.asin(y / 40075000))
    lat = originlat + deltaLat

    latcor = m.cos(m.radians(lat))
    deltaLon = m.degrees(m.asin(x / (40075000 * latcor)))
    lon = originlon + deltaLon

    return [lat, lon]


def mean_angle(deg):
    try:
        return AngleNormalize(round(m.degrees(m.phase(sum(m.rect(1, m.radians(d)) for d in deg) / len(deg))), 2))
    except:
        return 0


def stripDate(date_in):
    return date_in.replace("-", "")


def getAPIBoats():
    res = requests.get("http://192.168.128.179:2021/api/v1/boat/boats")  # http://192.168.128.179:2021/api/v1/boat/boats
    return res.json()


def getAPIDates(boat):
    res = requests.get(
        "http://192.168.128.179:2021/api/v1/boat/dates?boat=" + boat)  # http://192.168.128.179:2021/api/v1/boat/dates?boat=
    dates = res.json()

    dates_list = []
    for date in dates:
        dates_list.append(getDate(date))

    return dates_list


def getAPISessions(date, boat):
    sessions_list = []

    date_str = stripDate(date)
    res = requests.get(
        "http://192.168.128.179:2021/api/v1/sessions?boat=" + boat + "&date=" + date_str + "&filter=""")  # http://192.168.128.179:2021/api/v1/sessions?boat=
    sessions = res.json()

    for session in sessions:
        name_str = str(session['name'])
        boat_str = str(session['boat'])
        url_str = str(session['url'])

        url = "http://192.168.128.179:2021/api/v1/sessions/open"  # http://192.168.128.179:2021/api/v1/sessions/open
        body = {"TabletMode": False, "Departments": None, "BoatType": None, "EventTypes": None, "EventTypeGroups": None,
                "Name": name_str, "Boat": boat_str, "URL": url_str}
        headers = {"content-type": "application/json"}

        res = requests.post(url, headers=headers, data=json.dumps(body), verify=False)
        sources = res.json()

        dc = {"date": date, "session": session, "sources": sources, "existing_channels": [], "existing_sources": []}
        sessions_list.append(dc)

    return sessions_list


def getAPIChannels(name_str, url_str, source_str):
    url = "http://192.168.128.179:2021/api/v1/channels"  # http://192.168.128.179:2021/api/v1/channels
    body = {"SessionName": name_str, "SessionURL": url_str, "DataSourceName": source_str}
    headers = {"content-type": "application/json"}

    res = requests.post(url, headers=headers, data=json.dumps(body), verify=True)

    channels = []
    if len(res.json()) > 0:
        for row in res.json():
            channel = row['name']
            channels.append(channel)

    return channels


def getAPIChannelSources(sessions_list, selected_channels):
    for session_info in sessions_list:
        name_str = str(session_info['session']['name'])
        url_str = str(session_info['session']['url'])

        existing_sources = []
        existing_channels = []

        session_sources = session_info['sources']
        for source in session_sources:
            url = "http://192.168.128.179:2021/api/v1/channels"  # http://192.168.128.179:2021/api/v1/channels
            body = {"SessionName": name_str, "SessionURL": url_str, "DataSourceName": source}
            headers = {"content-type": "application/json"}

            res = requests.post(url, headers=headers, data=json.dumps(body), verify=True)
            channels = res.json()

            for req_channel in selected_channels:
                if list(filter(lambda channel: channel['name'] == req_channel, channels)) != []:
                    existing_sources.append(source)
                    existing_channels.append(req_channel)

        session_info['existing_channels'] = existing_channels
        session_info['existing_sources'] = existing_sources

    return sessions_list

def getAPIChannelValues(boat, date, channels, rs, starttime_str, endtime_str):
    # try:
    format_str = "%Y-%m-%d %H:%M:%S.%f"
    ref_datetime_str = date + " 00:00:00.000"
    datetime_ref = dt.strptime(ref_datetime_str, format_str)

    sessions_list = getAPISessions(date, boat)

    if len(sessions_list) > 0:
        updated_sessions_list = getAPIChannelSources(sessions_list, channels)

        if len(updated_sessions_list) > 0:
            starttime = dt.strptime(starttime_str, format_str)
            endtime = dt.strptime(endtime_str, format_str)

            starttime_sec = (starttime - datetime_ref).total_seconds()
            endtime_sec = (endtime - datetime_ref).total_seconds()

            session_frames = []
            for session_info in updated_sessions_list:
                session_url = session_info['session']['url']
                session_sources = session_info['sources']
                existing_sources = session_info['existing_sources']
                existing_channels = session_info['existing_channels']

                source_frames = []
                for source in session_sources:
                    ci = 0
                    frames = []
                    final_channels = []

                    for channel in existing_channels:
                        existing_source = existing_sources[ci]
                        ci += 1

                        if existing_source == source:
                            url = "http://192.168.128.179:2021/api/v1/channels/channelsValuesInterval"  # http://192.168.128.179:2021/api/v1/channels/channelsValuesInterval
                            headers = {"content-type": "application/json"}

                            try:
                                body = {"SessionURL": session_url, "IdSourceName": source,
                                        "TimeChannelName": "SystemTime_DaySeconds",
                                        "ChannelsName": ["SystemTime_DaySeconds", channel],
                                        "StartSeconds": str(starttime_sec), "EndSeconds": str(endtime_sec)}
                                res = requests.post(url, headers=headers, data=json.dumps(body), verify=True)
                                df = pd.DataFrame(res.json())
                                frames.append(df)
                            except:
                                df = pd.DataFrame()
                                frames.append(df)
                                print("Error Encountered:" + channel)

                            final_channels.append(channel)

                    if len(frames) > 0:
                        df = pd.concat(frames, axis=1)
                        dff = df.loc[:, ~df.columns.duplicated()]
                        dfd = pd.DataFrame([])
                        for col in dff.columns:
                            dfd = dff[~dff[col].isin(["NaN"])]
                            break

                        datetime_str = session_info['date'] + " 00:00:00.000"

                        format_str = "%Y-%m-%d %H:%M:%S.%f"
                        datetime_obj = dt.strptime(datetime_str, format_str)

                        dfdd = dfd.assign(
                            Datetime=datetime_obj + pd.to_timedelta(dfd["SystemTime_DaySeconds"], unit='s'))
                        dfi = dfdd.set_index('Datetime')

                        for channel in final_channels:
                            dfi[channel] = dfi[channel].astype('float')

                        dfs = dfi.resample(rs).min()
                        source_frames.append(dfs)

                dfi = pd.DataFrame()
                if len(source_frames) > 0:
                    dfn = pd.concat(source_frames, axis=1)
                    dfnd = dfn.drop(['SystemTime_DaySeconds'], axis=1)

                    dfff = dfnd.loc[:, ~dfnd.columns.duplicated()]
                    dfi = dfff.interpolate(method='time')
                    dfif = dfi.loc[(dfi.index >= starttime) & (dfi.index <= endtime)]
                    session_frames.append(dfif)

            dfss = pd.DataFrame()
            if len(session_frames) > 0:
                dfs = pd.concat(session_frames)
                dfs["Datetime"] = dfs.index
                cols = list(dfs.columns)
                cols = [cols[-1]] + cols[:-1]
                dfss = dfs[cols]

            return dfss

        dfss = pd.DataFrame()
        return dfss

    dfss = pd.DataFrame()
    return dfss


# except:
#     dfss = pd.DataFrame()
#     return dfss

def callGCP(host, queryType, options):
    # host = "ws://localhost:8080/"
    ws = create_connection(host)

    id = 0
    options["ID"] = id
    options["QueryType"] = queryType
    ws.send(json.dumps(options))
    while True:
        resultStr = ws.recv()
        result = json.loads(resultStr)
        if "ID" in result and result["ID"] == id:
            break
    if "Error" in result:
        raise RuntimeError(result["Error"])
    else:
        return result["Reply"]

    ws.close


def getGCPRunList(host):
    return callGCP(host, "RunList", {})


def getGCPChannelList(host, uuid):
    return callGCP(host, "RunChannelNames", {"UUID": uuid})


def getGCPChannelValues(host, uuid, channels):
    return callGCP(host, "RunChannelValues", {"UUID": uuid, "ChannelNames": channels})


def SetPeriodAvailableDates(giorni,
                            boat):  # Extract all available dates based on boat within a certain time period given with giorni
    # giorni: list of list. [[year,month,from,to],[year,month,from,to]] example: available dates of AC 75 between 02-09-2021 and 03-17-2021 -> ([[2021,2,2,28],[2021,3,1,17], 'AC 75')
    list = []
    for i in giorni:
        if i[1] < 10:
            for j in range(i[2], i[3] + 1):
                if j < 10:
                    list.append(str(i[0]) + '-' + '0' + str(i[1]) + '-0' + str(j))
                else:
                    list.append(str(i[0]) + '-' + '0' + str(i[1]) + '-' + str(j))
        else:
            for j in range(i[2], i[3] + 1):
                if j < 10:
                    list.append(str(i[0]) + '-' + str(i[1]) + '-0' + str(j))
                else:
                    list.append(str(i[0]) + '-' + str(i[1]) + '-' + str(j))

    dates = getAPIDates(boat)
    print(dates)
    final = [value for value in list if value in dates]
    final.reverse()
    return final


def SetSailingSessions(dates, boat, ses_to_take=['Sessions01'], ses_to_remove=['Dock'],
                       mode='remove'):  ## returns sessions on dates removing the one you don't want to analize. NOTE: easy to extend to other cases.
    final = []
    if mode == 'remove':
        for i in dates:
            sessions = getAPISessions(i, boat)
            for j in sessions:
                for j['session']['name'] in ses_to_remove:
                    sessions.remove(j)
            final.extend(sessions)
            sessions.clear()
    else:
        for i in dates:
            new = []
            sessions = getAPISessions(i, boat)
            for j in sessions:
                for j['session']['name'] in ses_to_take:
                    new.append(j)
            final.extend(new)

    return final


def Dataframes_train_test(dates_train, dates_test, dates, port_list, stbd_list, Boat='AC 75', frame_ms_tr='100ms',
                          frame_ms_te='100ms', begin_time='00:18:41.600000',
                          end_time='23:18:41.600000', min_Bs=25, Twa_query='35 < Twa < 46 or -45 < Twa < -35',
                          corr_m_path='CorrMatrix', data_path='Data'):
    '''''
    dates_train, dates_test: available dates for Boat
    dates
    port_list: port channel to retrive. Remember to also add to list targets channels, Twa, FCS_StdbCant_Ang and non-port non-stbd channels
    stbd_list: see above. The same but for starboard
    Boat: self explanatory
    frame_ms_tr and ts: frame rate for data. Note that is possible to use different frame rate to fight overfitting
    begin and end_time: self explanatory
    min_Bs: minimum boat speed. use for extrapolating good data
    Twa_query: see default to understand the form. Query to determine boat direction with respect to the wind e.g. upwind (default)
    corr_m_path, data_path: paths to store corr matrices and all dataframes. it automatically adjust to differentiate between port and stbd, train and test
    '''''
    if len(dates_train) != 0:
        df_list_port = []
        df_list_stbd = []
        for date in dates_train:
            print(date)  # To keep track of the process (can be very long)
            dati_port_train = getAPIChannelValues(Boat, date, port_list, frame_ms_tr, date + ' ' + begin_time,
                                                  date + ' ' + end_time)
            dati_port_train = dati_port_train[
                (dati_port_train['FCS_StbdCant_Ang'] > 60) & (dati_port_train['Bs'] > min_Bs)]
            dati_port_train = dati_port_train.query(Twa_query)

            dati_stbd_train = getAPIChannelValues(Boat, date, stbd_list, frame_ms_tr, date + ' ' + begin_time,
                                                  date + ' ' + end_time)
            dati_stbd_train = dati_stbd_train[
                (dati_stbd_train['FCS_PortCant_Ang'] > 60) & (dati_stbd_train['Bs'] > min_Bs)]
            dati_stbd_train = dati_stbd_train.query(Twa_query)

            df_list_port.append(dati_port_train)
            df_list_stbd.append(dati_stbd_train)

        df_port_train = pd.concat(df_list_port, axis=0)
        df_stbd_train = pd.concat(df_list_stbd, axis=0)

        df_port_train = df_port_train[port_list]
        df_stbd_train = df_stbd_train[stbd_list]

        if corr_m_path != None:
            st.CorrelationMatrix(df_port_train, save=True, path=corr_m_path + 'Port' + 'Train' + '.csv')
            st.CorrelationMatrix(df_stbd_train, path=corr_m_path + 'Stbd' + 'Train' + '.csv')

    ########### test
    if len(dates_test) != 0:
        df_list_port = []
        df_list_stbd = []
        for date in dates_test:
            print(date)
            dati_port_test = getAPIChannelValues(Boat, date, port_list, frame_ms_te, date + ' ' + begin_time,
                                                 date + ' ' + end_time)
            dati_port_test = dati_port_test[(dati_port_test['FCS_StbdCant_Ang'] > 60) & (dati_port_test['Bs'] > min_Bs)]
            dati_port_test = dati_port_test.query(Twa_query)

            dati_stbd_test = getAPIChannelValues(Boat, date, stbd_list, frame_ms_te, date + ' ' + begin_time,
                                                 date + ' ' + end_time)
            dati_stbd_test = dati_stbd_test[(dati_stbd_test['FCS_PortCant_Ang'] > 60) & (dati_stbd_test['Bs'] > min_Bs)]
            dati_stbd_test = dati_stbd_test.query(Twa_query)

            df_list_port.append(dati_port_test)
            df_list_stbd.append(dati_stbd_test)
        df_port_test = pd.concat(df_list_port, axis=0)
        df_stbd_test = pd.concat(df_list_stbd, axis=0)

        df_port_test = df_port_test[port_list]
        df_stbd_test = df_stbd_test[stbd_list]

        if corr_m_path != None:
            st.CorrelationMatrix(df_port_test, path=corr_m_path + 'Port' + 'Test' + '.csv')
            st.CorrelationMatrix(df_stbd_test, path=corr_m_path + 'Stbd' + 'Test' + '.csv')

        ##### save csv ####
        df_port_train.to_csv(data_path + 'train_port.csv')
        df_stbd_train.to_csv(data_path + 'train_stbd.csv')

        df_port_test.to_csv(data_path + 'test_port.csv')
        df_stbd_test.to_csv(data_path + 'test_stbd.csv')

    if len(dates) != 0:  # enters here if you want to make a data study without train and test.
        df_list = []
        for date in dates:
            print('eccoci')
            print(date)
            dati = getAPIChannelValues('AC 75', date, port_list, frame_ms_te, date + ' ' + begin_time,
                                       date + ' ' + end_time)
            dati = dati[(dati['FCS_StbdCant_Ang'] > 60) & (dati['Bs'] > min_Bs)]
            dati = dati.query(Twa_query)

            df_list.append(dati)
        df = pd.concat(df_list, axis=0)

        df = df[port_list]

        if corr_m_path != None:
            st.CorrelationMatrix(df, path=corr_m_path + '.csv')

        ##### save csv ####
        df.to_csv(data_path + 'data.csv')