'''
This class' objetive is giving an easy way to perform fast EDA.
'''
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#Load dataframe
df = pd.read_pickle('..\\..\\data\\processed\\regressor_to_matricula_SMART.pkl')


#The classes are unbalanced, so we have to balance them to extract info of the correlation

exito = df[df['Result']==True].copy()
exito['Conversion'] = 'Exito'
fracaso = df[df['Result']==False].copy()
fracaso['Conversion'] = 'Fracaso'
fracaso = fracaso.iloc[:exito.shape[0], :]
smart = pd.concat([exito, fracaso], ignore_index=True)
smart['dummy'] = 'Dummy'
print(smart.shape)

scaler = StandardScaler()
scaler.fit(smart['days_since_modification'].values.reshape(-1, 1))
smart['days_since_modification'] = scaler.transform(smart['days_since_modification'].values.reshape(-1, 1))
smart['days_since_modification'].hist()
plt.show()

'''
# Good plot to see the correlation of continuous variables
sns.swarmplot( x='dummy',y="days_since_modification", hue="Conversion",
              palette=["r", "c", "y"], data=smart)
plt.show()

#sns.distplot(exito['bi_score_c'])
#sns.distplot(fracaso['bi_score_c'])



top_values = smart.prkt_ciudades_id_c.value_counts()
top_values = top_values[:10].index
smart['ciudad'] = smart.prkt_ciudades_id_c.where(smart.prkt_ciudades_id_c.isin(top_values),'Other')
'''
# Plot for histograms
sns.catplot(x="rating",kind = 'count', palette="bright", data=smart,hue='Result');
plt.xlabel('Valor de la característica rating')
plt.ylabel('Nº de registros')
plt.legend()
plt.title('Comparativa de Exitos/Fracasos según rating')
plt.show()

'''
target_0 = smart.loc[smart['Result'] == True]
target_1 = smart.loc[smart['Result'] == False]


sns.distplot(target_0[['days_since_modification']], hist=False, rug=False,label='Exito')
sns.distplot(target_1[['days_since_modification']], hist=False, rug=False, label='Fracaso')
plt.xlabel('Dias desde la ultima modificación')
plt.title('Distribución de probabilidad')
plt.show()

corrmat = df.corr()
top_corr_features = corrmat.index
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()'''