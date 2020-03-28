
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt


#--- lambda function to parse date from string format
parser = lambda x: dt.datetime.strptime(x,'%Y-%m-%dT%H:%M:%S.%fZ')
#--- Load buys
buys = pd.read_csv('yoochoose-buys.dat',
                   dtype={0: int,1:np.str_,2:int,3:int,4:np.int8}, #specify col dtypes for less memory consumption 
                   parse_dates=[1], #define which col has dates
                   date_parser=parser, # employ lambda function
                   names=['SessionId','Timestamp','ItemId','Price','Purchase'],
                   header=None)
#--- Drop price column
buys = buys.drop(['Price'],1)
#--- Load clicks
clicks = pd.read_csv('yoochoose-clicks.dat',
                   dtype={0: int,1:np.str_,2:int,3:object},
                   parse_dates=[1], 
                   date_parser=parser,
                   names=['SessionId','Timestamp','ItemId','Category'],
                   header=None)
#--- Drop CAtegory column
clicks=clicks.drop('Category',1)
#--- Append the two dataframes
sessions = clicks.append(buys, ignore_index=True)
#--- REplace nan values with zero
sessions['Purchase'] = sessions['Purchase'].fillna(0)
#--- Find unique items per session
clickedItems = sessions.groupby('SessionId').ItemId.nunique()
#--- Reset index for appending later
clickedItems = clickedItems.reset_index()
#--- Aggregate function on df to get:
#------------ 1.when the session started
#------------ 2.how many clicks in session
#------------ 3.if there was a buy or not
sessions = sessions.groupby(['SessionId']).agg({'Timestamp' : 'min', 'ItemId' : 'count','Purchase':'sum'})
#--- reset index od df
sessions = sessions.reset_index()
#--- rename column
sessions['clickedItems'] = clickedItems['ItemId']

#--- construct function to grab weekday and hour the session started
def getWeekAndHour(df,columnName):
    weekday=[]
    hour=[]
    for i in range(sessions.shape[0]):
        weekday.append(df[columnName][i].weekday())
        hour.append(df[columnName][i].hour)
    timeDf = pd.DataFrame({'Weekday': weekday,'Hour': hour})
    return timeDf
#--- concatenate df with new columns Weekday and Hour
sessions = pd.concat([sessions,getWeekAndHour(sessions,'Timestamp')],axis=1)
#--- drop timestamp column
sessions = sessions.drop('Timestamp',1)
#--- rename column itemId to clicks
sessions = sessions.rename(columns={"ItemId": "Clicks"})
#--- rearrange columns to move class column ('Purchase') to the end
sessions = sessions[['SessionId', 'Clicks', 'clickedItems', 'Weekday','Hour','Purchase']]
#--- no buy:0 //  buy:1
sessions['Purchase'] = (sessions['Purchase'] !=0).astype(np.int8)
#--- Fun time: plot hour,weekday vs buying rate
#--- this shows us which hours and weekdays are more active 
fig, axs = plt.subplots(1, 2, figsize=(10,4))
#fig.subplots_adjust(hspace=.01)
buyRateHour = sessions.groupby('Hour')['Purchase'].mean()
buyRateHour.sort_index().plot(ax=axs[0], grid=True, linewidth=2, ylim=(0.01, 0.035))
axs[0].set_ylabel('Buying rate')
axs[0].set_xlabel('Hour')
#--- this dictionary is to transform from 0,1,2... to Mon,Tue,Wed...
WEEKDAY_TO_NAME = {
    0: 'Mon',
    1: 'Tue',
    2: 'Wed',
    3: 'Thu',
    4: 'Fri',
    5: 'Sat',
    6: 'Sun',
}

buyRateDay = sessions.groupby('Weekday')['Purchase'].mean()
buyRateDay = buyRateDay.sort_index().reset_index()
buyRateDay['Weekday'] = buyRateDay['Weekday'].map(lambda n: WEEKDAY_TO_NAME[n])
buyRateDay = buyRateDay.set_index('Weekday')
buyRateDay.plot(ax=axs[1], kind='bar', legend=False, ylim=(0.01, 0.035), grid=True, sharey=True)

plt.savefig('buyRate.png', bbox_inches='tight')






