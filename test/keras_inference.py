# LOAD MODULES
import onnxruntime as rt
import numpy as np
import pandas as pd


# SET SEED
np.random.seed(92)

# LOAD MODEL AND PREPROCESSING DATASET
sess = rt.InferenceSession("..\\model\\conversion_probability.onnx")
df = pd.read_pickle('..\\data\\processed\\regressor_to_matricula_SMART_complete.pkl')

# DROP NON USED FEATURES
labels = ['idcurso','nivel_educativo', 'ciudad','tipo']
features = ['days_since_modification','idcurso','nivel_educativo', 'ciudad', 'tipo']
df = df[features]

# POSSIBLE VALUES AND DEFAULT VALUES
input_list = [1,'19920', 'Ingeniero', 'EC', 'otra cosas']
for index, label in enumerate(labels):
    label_possible_values = df[label].unique().tolist()
    if not(input_list[index] in label_possible_values):
        input_list[index + 1] = df[label].mode()[0]

# INPUT (SUSTITUIR DF.ILOC[1] POR EL ARRAY DE ENTRADA REAL)
df.loc['inference'] = input_list
'''      
idcurso_possible_values = df.idcurso.unique().tolist()
tipo_possible_values = df.tipo.unique().tolist()
ciudad_possible_values = df['ciudad'].unique().tolist()
rating_possible_values = df.rating.unique().tolist()
education_level_possible_values = df.education_level_c.unique().tolist()

idcurso_default_value = df.idcurso.mode()[0]
tipo_default_value = df.tipo.mode()[0]
ciudad_default_value = df.ciudad.mode()[0]
education_level_default = df.education_level_c.mode()[0]
rating_default = df.rating.mode()[0]
'''


for label in labels:
    one_hot = pd.get_dummies(df[label], prefix=label)
    # Drop column B as it is now encoded
    df = df.drop(label, axis=1)
    # Join the encoded df
    df = df.join(one_hot)


input_name = sess.get_inputs()[0].name

# ESTOS DOS NUMEROS DEBEN DE SER IGUALES
print('Next 2 numbers have to be equal:')
print('Number of input nodes of the first layer: ' +str(sess.get_inputs()[0].shape[1]))
print('Number of columns after OHE: ' + str(df.shape[1]))


label_name = 'dense_3_Sigmoid_0'

X_test = np.array(df.loc['inference'])
print(len(X_test))
pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})

print(np.round(pred_onx[0],decimals=2))

