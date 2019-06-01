import  seaborn as sns
import  pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({"a":[1, 2, 3, 4], "b":[5, 5, 5, 6],"c":[53, 26, 17, 18]})
print(df)

sns.pairplot(df,plot_kws=dict(alpha=0.05))
plt.show()