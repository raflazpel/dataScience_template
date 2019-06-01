import pandas as pd

print(pd.get_option('display.encoding'))
data = pd.read_json('data.json')
data.describe()
