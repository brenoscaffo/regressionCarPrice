# Bibliotecas usadas no treino
#%%
import lightgbm as lgbm
import xgboost as xgb
import mlflow
import sklearn
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

#%%
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment(experiment_id="4")
# No meu ambiente do MLFlow é o experimento 3, verifique qual o ID do experimento

# Algoritmo de Regressao Linear
#%%
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, 
                                                    test_size=0.33, random_state=42)

#%%
model_dummies = LinearRegression()
model_dummies.fit(X_train, y_train)

#%%
y_pred = model_dummies.predict(X_test)
y_pred

#%%
y_test

#%%
r2 = r2_score(y_test, y_pred)
r2

#%%
n = X_test.shape[0]
p = X_test.shape[1]

r2_adjust = 1 - ((1-r2)*(n-1))/(n-p-1)
r2_adjust

#%%
mae = mean_absolute_error(y_test, y_pred)

#%%
rmse = root_mean_squared_error(y_test, y_pred)

#%%
mape = mean_absolute_percentage_error(y_test, y_pred)

#%%
print(f'Mean absolute error: {np.round(mae, 2)}')
print(f'Root mean squared error: {np.round(rmse,2)}')
print(f'Mean absolute percentage error: {np.round(100*mape,2)}')
print(f'R2: {np.round(r2,4)}')
print(f'R2 adjust: {np.round(r2_adjust,4)}')

# Usando Label Encoding
#%%
X_train_label, X_test_label, y_train, y_test = train_test_split(Xlabel, y, 
                                                    test_size=0.33, random_state=42)

#%%
model_label = LinearRegression()
model_label.fit(X_train_label, y_train)

#%%
y_pred = model_label.predict(X_test_label)
y_pred

#%%
y_test

#%%
r2_label = r2_score(y_test, y_pred)
r2_label

#%%
n = X_test_label.shape[0]
p = X_test_label.shape[1]

r2_adjust_label = 1 - ((1-r2_label)*(n-1))/(n-p-1)
r2_adjust_label

#%%
mae_label = mean_absolute_error(y_test, y_pred)

#%%
rmse_label = root_mean_squared_error(y_test, y_pred)

#%%
mape_label = mean_absolute_percentage_error(y_test, y_pred)

#%%
print(f'Mean absolute error: {np.round(mae_label, 2)}')
print(f'Root mean squared error: {np.round(rmse_label,2)}')
print(f'Mean absolute percentage error: {np.round(100*mape_label,2)}')
print(f'R2: {np.round(r2_label,4)}')
print(f'R2 adjust: {np.round(r2_adjust_label,4)}')

# Algoritmos baseados em arvores
# Teste para todas as features e para as mais importantes para o modelo
# Treino na tabela das dummies
# Construçao do tunning dos hiperparametros
#%%
modelos = {
    "Decision_Tree": {
        "model": DecisionTreeRegressor(random_state=42),
        "params":{"max_depth": [10,15,20,25,30,40,50],
                "min_samples_split": [2, 3, 4, 5, 8, 10],
                "min_samples_leaf": [1, 2, 2.5, 3, 5, 6, 6.5],
                "ccp_alpha": [0, 0.0001, 0.00001, 0.0005]}        
    },

    "XGBoost": {
        "model": xgb.XGBRegressor(random_state=42),
        "params": {
            "max_depth": [5, 8, 10, 15, 20],
            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
            "n_estimators": [100, 200, 300, 500, 750, 1000],
            "subsample": [0.6, 0.8, 0.9, 0.99]
        }
    },

    "LightGBM": {
        "model": lgbm.LGBMRegressor(random_state=42),
        "params": {
            "max_depth": [5, 8, 10, 15, 20],
            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
            "n_estimators": [100, 200, 300, 500, 750, 1000],
            "num_leaves": [31, 40, 50]
        }
    }
}

#%%
for nome, info in modelos.items():
    with mlflow.start_run(run_name=nome):

        grid = GridSearchCV(
            estimator=info['model'],
            param_grid=info['params'],
            cv = 5,
            verbose=4,
            scoring="neg_mean_absolute_error"
        )

        grid.fit(X_train, y_train)

        #melhor modelo
        best_model = grid.best_estimator_
        # melhor modelo
        pred = best_model.predict(X_test)

        
        # =======================
        # Predições treino
        # =======================
        
        y_train_pred = best_model.predict(X_train)
        
        r2_train = metrics.r2_score(y_train, y_train_pred)
        mae_train = metrics.mean_absolute_error(y_train, y_train_pred)
        rmse_train = metrics.root_mean_squared_error(y_train, y_train_pred)
        mape_train = metrics.mean_absolute_percentage_error(y_train, y_train_pred)

        
        # =======================
        # Predições teste
        # =======================

        y_test_pred = best_model.predict(X_test)

        r2_test_dummies = metrics.r2_score(y_test, y_test_pred)
        mae_test_dummies = metrics.mean_absolute_error(y_test, y_test_pred)
        rmse_test_dummies = metrics.root_mean_squared_error(y_test, y_test_pred)
        mape_test_dummies = metrics.mean_absolute_percentage_error(y_test, y_test_pred)
        # registro de parametros
        mlflow.log_params(grid.best_params_)

        # registro de metricas
        mlflow.log_metrics({
            "R2_Teste": r2_test_dummies,
            "MAE_Teste": mae_test_dummies,
            "RMSE_Teste": rmse_test_dummies,
            "MAPE_Teste": mape_test_dummies
        })

        print(f'{nome} finalizado!')

#%%
r2_adj_test_dummies = 1-((1-r2_test_dummies)*(X_test.shape[0]-1))/(X_test.shape[0]-X_test.shape[1]-1)


# Metricas do melhor modelo
#%%
print(f'O R2 desse modelo é: {r2_test_dummies}')
print(f'O R2 ajustado desse modelo é: {r2_adj_test_dummies}')
print(f'O Erro Médio Absoluto desse modelo é: {mae_test_dummies}')
print(f'A raíz quadrado do erro médio  desse modelo é: {rmse_test_dummies}')
print(f'O percentual do erro médio absoluto desse modelo é: {mape_test_dummies}')

#Verifique as metricas que estao no servidor local do mlflow para comparar o desempenhos dos demais algoritmos
# Treino da tabela com LabelEncoder

modelos_label = {
    "Decision_Tree_Label": {
        "model": DecisionTreeRegressor(random_state=42),
        "params":{"max_depth": [10,15,20,25,30,40,50],
                "min_samples_split": [2, 3, 4, 5, 8, 10],
                "min_samples_leaf": [1, 2, 2.5, 3, 5, 6, 6.5],
                "ccp_alpha": [0, 0.0001, 0.00001, 0.0005]}        
    },

    "XGBoost_Label": {
        "model": xgb.XGBRegressor(random_state=42),
        "params": {
            "max_depth": [5, 8, 10, 15, 20],
            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
            "n_estimators": [100, 200, 300, 500, 750, 1000],
            "subsample": [0.6, 0.8, 0.9, 0.99]
        }
    },

    "LightGBM_Label": {
        "model": lgbm.LGBMRegressor(random_state=42),
        "params": {
            "max_depth": [5, 8, 10, 15, 20],
            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
            "n_estimators": [100, 200, 300, 500, 750, 1000],
            "num_leaves": [31, 40, 50]
        }
    }
}

#%%
for nome, info in modelos_label.items():
    with mlflow.start_run(run_name=nome):

        grid = GridSearchCV(
            estimator=info['model'],
            param_grid=info['params'],
            cv = 5,
            verbose=4,
            scoring="neg_mean_absolute_error"
        )

        grid.fit(X_train_label, y_train)

        #melhor modelo
        best_model = grid.best_estimator_
        # melhor modelo
        pred = best_model.predict(X_test_label)

        
        # =======================
        # Predições treino
        # =======================
        
        y_train_pred = best_model.predict(X_train_label)
        
        r2_train = metrics.r2_score(y_train, y_train_pred)
        mae_train = metrics.mean_absolute_error(y_train, y_train_pred)
        rmse_train = metrics.root_mean_squared_error(y_train, y_train_pred)
        mape_train = metrics.mean_absolute_percentage_error(y_train, y_train_pred)

        
        # =======================
        # Predições teste
        # =======================

        y_test_pred = best_model.predict(X_test_label)

        r2_test_label = metrics.r2_score(y_test, y_test_pred)
        mae_test_label = metrics.mean_absolute_error(y_test, y_test_pred)
        rmse_test_label = metrics.root_mean_squared_error(y_test, y_test_pred)
        mape_test_label = metrics.mean_absolute_percentage_error(y_test, y_test_pred)
        # registro de parametros
        mlflow.log_params(grid.best_params_)

        # registro de metricas
        mlflow.log_metrics({
            "R2_Teste": r2_test_label,
            "MAE_Teste": mae_test_label,
            "RMSE_Teste": rmse_test_label,
            "MAPE_Teste": mape_test_label
        })

        print(f'{nome} finalizado!')

#%%
r2_adj_test_label = 1-((1-r2_test_label)*(X_test_label.shape[0]-1))/(X_test_label.shape[0]-X_test_label.shape[1]-1)

# Metricas do melhor modelo
#%%
print(f'O R2 desse modelo é: {r2_test_label}')
print(f'O R2 ajustado desse modelo é: {r2_adj_test_label}')
print(f'O Erro Médio Absoluto desse modelo é: {mae_test_label}')
print(f'A raíz quadrado do erro médio  desse modelo é: {rmse_test_label}')
print(f'O percentual do erro médio absoluto desse modelo é: {mape_test_label}')

# Versaos usadas no estudo
#%%
print(f'MLflow: {mlflow.__version__}')
print(f'Scikit-learn: {sklearn.__version__}')
print(f'XGBoost: {xgb.__version__}')
print(f'LightGBM: {lgbm.__version__}')