'''
This class intends to download the information from the 'Oportunities' dataset, perform EDA and data cleaning and load
the result in a new dataframe for further exploration
'''

# Import values
import sqlalchemy as db
import pandas as pd
# IMPORT  MODULES
import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
np.random.seed(92)
from numpy.random import seed
seed(92)
import time
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
import flask
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import StandardScaler
'''
Configuration of the class
'''
# Pickle's name where results are saved
pickle_file_name = 'regressor_to_matricula_SMART_complete.pkl'
csv_file_name = 'regressor_to_matricula_production.csv'
pickle_production_file_name = 'regressor_to_matricula_production.pkl'
# Database settings


# Datetime structure
datetime_structure = '%Y-%m-%dT%H:%M:%S'

# Features saved into the smart model
smart_labels = ['idcurso', 'nivel_educativo', 'ciudad', 'tipo', 'days_since_modification', 'Result']
'''
First block: Load data
'''
# Create connection to database
engine = db.create_engine(database_name)

# Load all info in table to 'dataframe'.
df = pd.read_sql_table(table_name, engine)

'''
Second block: FE. Find the 99 percentile for the users that do not convert 
'''


def slice_time(s):
    '''
    :param s:
    :return:
    This function translates the datamart String timestamp to a datetime object
    '''
    s = s[0:19]
    date_time_obj = datetime.datetime.strptime(s, datetime_structure)
    return date_time_obj


# Select the ones with 'prematricula' to study the distribution
matriculas = df[df['status'] == 'matricula_realizada'].copy()

# TODO La nueva API no trae valores de segundo nivel del JSON. Por tanto no tenemos 'prematri_fechahora' ahora mismo
'''
matriculas.loc[:,'prematri_fechahora'] = matriculas['prematri_fechahora'].map(slice_time)
matriculas['date_entered_utc'] = matriculas['date_entered_utc'].map(slice_time)
matriculas['time_to_matricula'] = matriculas['prematri_fechahora'] - matriculas['date_entered_utc']
matriculas['time_to_matricula'].dt.total_seconds().div(3600*24).hist()
plt.xlabel("Time (days) ")
plt.ylabel("Number of conversions")
plt.title("Distribution of conversions for 3 months")
plt.show()
'''
# TODO Automatizar los 55 dias cuando esté disponible la prematri_fechahora
# time_window_conversion = matriculas.time_to_matricula.quantile(0.99)
time_window_conversion = datetime.timedelta(days=55)
'''
Third block: Feature Engineering & Data Cleaning.
'''

# Calculate and add the target variable.
leads = df[df['status'] != 'matricula_realizada'].copy()
leads['date_entered_utc'] = leads['date_entered_utc'].map(slice_time)
# TODO This 'now' should be automated. Cuando esté disponible la fecha de carga de los datos sustituirla por el hardcode
now = datetime.datetime(2019, 5, 16, 23, 59, 59, 99)
leads['time_passed_since_lead'] = leads['date_entered_utc'].apply(lambda x: now - x)

# Only allow in the dataframe user that have been more days than the 99 percentile of time to convert.
leads = leads[leads.time_passed_since_lead.dt.total_seconds().div(3600*24) > time_window_conversion.days]

# Create target feature by joining datasets
leads['Result'] = False
matriculas['Result'] = True
smart = pd.concat([leads, matriculas], ignore_index=True, sort=False)

# CREATE TIME PASSED SINCE LAST MODIFICATION
smart.date_modified_utc.fillna(method='ffill', inplace=True)
smart['date_modified_utc'] = smart['date_modified_utc'].map(slice_time)
smart['date_modified_utc'] = smart['date_modified_utc'].apply(lambda x: now - x)
smart['days_since_modification'] = smart['date_modified_utc'].dt.total_seconds().div(3600*24)

# Clean 'ciudad'
smart.prkt_ciudades_id_c.fillna(method='ffill', inplace=True)
smart['prkt_ciudades_id_c'] = smart['prkt_ciudades_id_c'].replace('','Empty')
top_values = smart.prkt_ciudades_id_c.value_counts()
top_values = top_values[:15].index
smart['ciudad'] = smart.prkt_ciudades_id_c.where(smart.prkt_ciudades_id_c.isin(top_values),'Other')

# Reduce number of categories 'education_level'
smart.education_level_c.fillna(method='ffill', inplace=True)
top_values = smart.education_level_c.value_counts()
top_values = top_values[:15].index
smart['nivel_educativo'] = smart.education_level_c.where(smart.education_level_c.isin(top_values),'Other')

# Reduce number of categories 'prkt_paises_id1_c'
smart.prkt_paises_id1_c.fillna(method='ffill', inplace=True)
top_values = smart.prkt_paises_id1_c.value_counts()
top_values = top_values[:15].index
smart['pais'] = smart.prkt_paises_id1_c.where(smart.prkt_paises_id1_c.isin(top_values),'Other')

'''
Fourth block: Feature selection
'''
# Set oppo_id as index
smart.set_index(smart['oppo_id'], inplace=True, drop=True)

# Deletion of fields that change over time.
smart.drop(labels=['prematri_esmatriculanuevoingreso','prematri_titulacionacceso_insti','prematri_titacceso_titulo',\
                'prematri_titacceso_nrosemestresrealizados','prematri_moneda_pago','prematri_moneda_base',\
                'prematri_formapagoidintegracion','prematri_ind_cambio','prematri_imp_base','prematri_importe_dto',\
                'estado_c','fases_c','date_modified','fase','sales_stage','prematri_fechahora'],\
                inplace=True, axis=1)


# Delete fields not correlated  (Based on business knowledge)
# TODO Esto se debería hacer filtrando los campos en la query.
# TODO ¿que es idcupon? ¿un id de la oportunidad alternativo?
# TODO probar usando datos de "visits" y de "clients"
smart.drop(labels=['idcupon','idvisitor','client_id','oppo_id','date_entered','date_entered_utc','first_contact'], inplace=True, axis=1)

# Deletion of fields with no variance
smart.drop(labels=['area','convocatoria_c'], inplace=True, axis=1)

# Deletion of fields with repeated info
smart.drop(labels=['name','nombrecurso','province_c','pais_de_residencia_c'], inplace=True, axis=1)

# Deletion of categorical fields with too few or too many classes
smart.drop(labels=['url','postal_code_c','education_level_c'], inplace=True, axis=1)

# Deletion of intermediate steps Feature engineering
smart.drop(labels=['time_passed_since_lead','date_modified_utc','prkt_paises_id1_c'], inplace=True, axis=1)
# smart.drop(labels=[ 'time_to_matricula'], inplace=True, axis=1)

# Deletion of fields not correlated. Based on EDA.
smart.drop(labels=['origen', 'phone_type', 'bi_score_c'], inplace=True, axis=1)

# Deletion of variables with too many nulls
smart.drop(labels=['prkt_ciudades_id_c'], inplace=True, axis=1)

# TODO Elegir que hacer cuando respondan sobre estas variables
smart.drop(labels=['reserva_c', 'descuento_c', 'price_c', 'pendiente_pago_c'], inplace=True, axis=1)


'''
Fourth block: Feature cleaning
'''

# Normalize non-categorical features

# Fill NA
smart.asesor_id.fillna(method='ffill', inplace=True)
smart.nivel_educativo.fillna(method='ffill', inplace=True)
smart.rating.fillna(method='ffill', inplace=True)
# smart.bi_score_c.fillna(smart.bi_score_c.median(),inplace = True)
smart.status.fillna(method='ffill', inplace=True)
# Convert to categorical
smart['asesor_id'] = smart.asesor_id.astype('category')
smart['nivel_educativo'] = smart.nivel_educativo.astype('category')
smart['idcurso'] = smart.idcurso.astype('category')
#smart['prkt_ciudades_id_c'] = smart.prkt_ciudades_id_c.astype('category')
smart['pais'] = smart.pais.astype('category')
smart['ciudad'] = smart.ciudad.astype('category')
smart['rating'] = smart.rating.astype('category')
smart['tipo'] = smart.tipo.astype('category')
smart['status'] = smart.status.astype('category')

# Random order for the dataframe
smart = smart.sample(frac=1)

# Save only useful labels for the smart model
smart = smart[smart_labels]

# Save dataframe to use in production
smart.to_pickle('..\\..\\data\\processed\\' + pickle_production_file_name)

# Standarize for traininig
scaler = StandardScaler()
scaler.fit(smart['days_since_modification'].values.reshape(-1, 1))
smart['days_since_modification'] = scaler.transform(smart['days_since_modification'].values.reshape(-1, 1))

# Save dataframe for training
smart.to_pickle('..\\..\\data\\processed\\' + pickle_file_name)

