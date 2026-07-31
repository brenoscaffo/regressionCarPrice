#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from kagglehub import KaggleDatasetAdapter

#%%
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment(experiment_id="3")
#%%

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "adhurimquku/ford-car-price-prediction",
    "ford.csv"
)

df.head()

#%%

df.shape

#%%

df.info()

#%%

df.describe()

#%%

df.isnull().sum()

#%%

sns.histplot(df['price'], bins = 50, kde = True)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Price distribuition")

#%%

sns.heatmap(df.corr(numeric_only = True), annot=True)
plt.yticks(rotation = 0)
plt.title("Correlation Heatmap")

#%%

sns.boxplot(data= df, x = 'year', y = 'price', palette='viridis')
plt.xticks(rotation = 90)
plt.xlabel("Year")
plt.ylabel("Price")
plt.title("Distribuition of Price by year")

#%%

sns.scatterplot(data = df, x = 'mileage', y = 'price')
plt.xlabel("Distance (miles)")
plt.ylabel("Price")

#%% 
sns.boxplot(data = df, x = 'engineSize', y = 'price', palette='viridis')
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.title("Distribution of Price by Engine Size")

#%%
sns.boxplot(data = df, x = 'transmission', y = 'price', palette='viridis')
plt.xlabel("Transmission")
plt.ylabel("Price")
plt.title("Distribution of Price by Transmission")


#%%
sns.boxplot(data = df, x = 'fuelType', y = 'price', palette='Set2')
plt.xlabel("Fuel Type")
plt.ylabel("Price")
plt.title("Distribution of Price by Fuel Type")


#%%

sns.boxplot(data = df, x = 'model', y = 'price', palette='viridis')
plt.xticks(rotation = 90)
plt.xlabel("Models")
plt.ylabel('Price')
plt.title('Box-Plot of sell price by models')


#%% 
df = df[df['year'] < 2060]
df

#%%
X = df.drop(columns = ['price'], axis = 1)
y = df['price']
X

#%%
df.columns
#%%
X_encoded = pd.get_dummies(X, columns = ['model', 'transmission', 'fuelType'], drop_first=True)

# Qual modelo irá perfomar melhor? De dummies ou LabelEncoder?
#%%
X_encoded = X_encoded.astype(int)
X_encoded

#%%
columns = ['model', 'transmission', 'fuelType']

Xlabel = X.copy()
label_encoder = {}

for i in columns:
    le = LabelEncoder()
    Xlabel[i] = le.fit_transform(Xlabel[i].astype(str))
    label_encoder[i] = le

Xlabel['model'].value_counts()
#%%
numeric_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
scaler = StandardScaler()

X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])
#%%
X_encoded

# Mesmo processo para o df com o processamento de LabelEncoder
#%% 
columns = Xlabel.columns
columns

#%%
Xlabel[columns] = scaler.fit_transform(Xlabel[columns])
Xlabel