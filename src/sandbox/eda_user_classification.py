# linear algebra and data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
# DDBB Connection libraries
import psycopg2
import psycopg2.extras
from matplotlib import pyplot
import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns
from pandas.io.sql import SQLTable
import matplotlib.pyplot as plt

def _execute_insert(self, conn, keys, data_iter):
    data = [dict((k, v) for k, v in zip(keys, row)) for row in data_iter]
    conn.execute(self.insert_statement().values(data))


SQLTable._execute_insert = _execute_insert
import importlib

importlib.reload(pd)

# Time libraries
import pytz
from pytz import timezone

import time, warnings
import datetime

warnings.filterwarnings("ignore")


# Arguments (x = value, p = recency, duration_value, frequency, d = quartiles dict)
def RScore(x, p, d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]:
        return 2
    else:
        return 1


# Arguments (x = value, p = recency, duration_value, frequency, d = quartiles dict)
def FDScore(x, p, d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4


# Time zone Madrid
Madrid = timezone('Europe/Madrid')
# Time zone Mexico
MexicoBajaNorte = timezone('Mexico/BajaNorte')
MexicoBajaSur = timezone('Mexico/BajaSur')
MexicoGeneral = timezone('Mexico/General')

# Loop attribute

loop1_attribute = 'AREA'
loop2_attribute = 'PRODUCTO'
loop3_attribute = 'ESTADO_VENTA'

# Internal attributes
vuser = 'USERNAME'
vtransacid = 'TRANSACID'
vtimestamp = 'TRANSACTIME'
vvisits = 'URL'
vduration = 'DURATION'

# Dataset row minimum
nrowmin = 10
# Initialize dataframe to None
quantiles_tot = None
rfm_segmentation_tot = None

# ====== Connection ======
# Connecting to IBM CLOUD DDBB
conn = psycopg2.connect(
    "sslmode=require host=sl-eu-de-1-portal.9.dblayer.com port=17015 dbname=compose user=admin password=ZMSQUQMQBBNVQTEU")
# ====== Connection ======
# Connecting to PostgreSQL by providing a sqlachemy engine
engine = create_engine('postgresql+psycopg2://admin:ZMSQUQMQBBNVQTEU@sl-eu-de-1-portal.9.dblayer.com:17015/compose')

# ====== Reading table =====#
# Reading PostgreSQL table into a pandas DataFrame
datasettot = pd.read_sql('select * from public."ETL_STG_TRANSAC_DATA_3M_UV"', engine)

# Create a filter in order to delete daily transactions
datasettot['date'] = pd.DatetimeIndex(datasettot[vtimestamp]).date
now = datetime.datetime.now(tz=Madrid).date()
date_filter = datasettot['date'] != now
datasettot = datasettot[date_filter].drop(['date'], axis=1)
# Create dataset with selected attributes
total_attribute = [loop1_attribute, loop2_attribute, loop3_attribute, vuser, vtransacid, vtimestamp, vvisits, vduration]
datasettot = pd.DataFrame(datasettot, columns=total_attribute)
# Create a filter AREA, PRODUCTO, ESTADO_VENTA table into a pandas DataFrame
dfilterrule = datasettot.drop_duplicates([loop1_attribute, loop2_attribute, loop3_attribute])[
    [loop1_attribute, loop2_attribute, loop3_attribute]]

indiloop = 0

total = pd.DataFrame(columns=['RECENCY','FREQUENCY','DURATION'])

for indexloop, rowloop in dfilterrule.iterrows():
    # Loop on filter AREA, PRODUCTO, ESTADO_VENTA
    item1 = rowloop[loop1_attribute]
    item2 = rowloop[loop2_attribute]
    item3 = rowloop[loop3_attribute]

    # Create variable with TRUE if loop1_attribute is
    loop1_filter = datasettot[loop1_attribute] == item1
    loop2_filter = datasettot[loop2_attribute] == item2
    loop3_filter = datasettot[loop3_attribute] == item3

    dataset = pd.DataFrame(columns=total_attribute)
    dataset = datasettot[loop1_filter & loop2_filter & loop3_filter]

    # Drop the rows where at least one element is missing.
    # dataset = dataset.dropna()
    dataset = dataset.dropna(subset=[vtransacid, vtimestamp, vduration])

    dataset.index = range(len(dataset))

    dataset = dataset.drop([loop1_attribute, loop2_attribute, loop3_attribute], axis=1)
    total_rows = dataset.shape[0]

    if dataset[vuser].nunique() > nrowmin:

        print(item1, item2, item3)
        print("Summary..")
        # Exploring the unique values of each attribute
        print("Number of User Connections: ", dataset[vtransacid].nunique())
        print("Number of Visits: ", dataset[vvisits].nunique())
        print("Number of Users:", dataset[vuser].nunique())

        # Recency
        # To calculate recency, we need to choose a date point from which we evaluate how many days ago was the customer's last purchase.
        # now = dt.date(2018,9,13)
        # now=datetime.datetime.now().date()

        # Create a new column called date which contains the date of invoice only
        dataset['date'] = pd.DatetimeIndex(dataset[vtimestamp]).date

        # Group by  and check last date of transaction
        recency_df = dataset.groupby(by=vuser)['date'].max().reset_index(name='lastconnectiondate')
        recency_df.columns = [vuser, 'lastconnectiondate']

        # Calculate recency
        recency_df['RECENCY'] = recency_df['lastconnectiondate'].apply(lambda x: (now - x).days)

        # Drop LastConnectionDate as we don't need it anymore
        recency_df.drop('lastconnectiondate', axis=1, inplace=True)

        # Frequency
        # Frequency helps us to know how many times a user connected from us. To do that we need to check how many connections are registered
        # by the same user.

        # Drop duplicates
        dataset_copy = dataset.copy()
        dataset_copy.drop_duplicates(subset=[vtransacid, vuser], keep="first", inplace=True)
        frequency_df = dataset_copy.groupby(by=[vuser])[vtransacid].count().reset_index(name='FREQUENCY')
        frequency_df.columns = [vuser, 'FREQUENCY']

        # Duration
        # Duration attribute answers the question: How many time did the customer spent over time?

        # To do that, first, we will create a new column total cost to have the total duration per connection.
        # Create column total duration
        duration_df = dataset.groupby(by=[vuser]).agg({vduration: 'sum'}).reset_index()
        duration_df.columns = [vuser, vduration]

        # Create RFD Table
        # Merge recency dataframe with frequency dataframe
        temp_df = recency_df.merge(frequency_df, on=vuser)

        # Merge with monetary dataframe to get a table with the 3 columns
        rfm_df = temp_df.merge(duration_df, on=vuser)
        # Use CustomerID as index
        rfm_df.set_index(vuser, inplace=True)

        total = pd.concat([total, rfm_df])
        if(indexloop>1000):
            break


total['RECENCY'] = pd.to_numeric(total['RECENCY'],downcast='float')
total['FREQUENCY'] = pd.to_numeric(total['FREQUENCY'],downcast='float')
sns.pairplot(total, plot_kws=dict(alpha=0.1))
ax = sns.distplot(total['DURATION'], bins=30, kde=False)
plt.show()

#fig = pyplot.figure()
#threeDax = Axes3D(fig)
#threeDax.scatter(total['RECENCY'], total['FREQUENCY'], total['DURATION'])
#pyplot.show()
