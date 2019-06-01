# Import libraries
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# Import data
pd.set_option('display.width', 500)
pd.set_option('display.max_columns',10)
df = pd.read_csv("..\\data\\opportunities.csv",names = ["Area", "Tipo", "Fase", "idvisitor","Curso","NumeroQueNoSeAQueCorresponde","date_modified","Pais","Precio","NombreCurso","date_entered","idCurso","hashemail","Estado","Id"],index_col= False)
df.set_index("Id",drop = True, inplace = True)



#Filter cases where estado = "matricula_realizada"

matricula_hecha = df[df['Estado'] == 'matricula_realizada']


# Visualize values
matricula_hecha['Fase'].value_counts().plot(kind='bar')
plt.show()

matricula_hecha['Estado'].value_counts().plot(kind='bar')
plt.show()
