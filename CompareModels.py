from matplotlib import pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score
import MachineLearning as utils
import DeepLearning as utilsDL
import pickle
import pandas as pd





def backtest(df):

    df['pred'] = df['pred'].shift(1)
    df['real'] = df['real'].shift(1)
    df.dropna(inplace=True)
    model_returns = df['pred'] * df['Close'].pct_change(1)


    # Calculate Sortino Ratio
    negative_model_returns = model_returns[model_returns < 0]
    downside_std = negative_model_returns.std()
    sortino_ratio = 100*np.sqrt(len(df))*(model_returns.mean() - 0) / downside_std

    # Calculate Alpha and Beta
    benchmark_returns = df['Close'].pct_change(1)
    benchmark_returns = benchmark_returns.fillna(0)
    
    # Calculate Sharpe Ratio
    risk_free_rate = 0.03/len(df)  # Assuming no risk-free rate
    excess_returns = model_returns - risk_free_rate
    sharpe_ratio = 100*np.sqrt(len(df))* excess_returns.mean() / excess_returns.std()
    
    
    # Calculate returns for correct and incorrect predictions
    negative_model_returns = model_returns[model_returns < 0]
    positive_model_returns = model_returns[model_returns > 0]
    neg_max = negative_model_returns.max()
    neg_min = negative_model_returns.min()
    neg_mean = negative_model_returns.mean()
    neg_std = negative_model_returns.std()
    pos_max = positive_model_returns.max()
    pos_min = positive_model_returns.min()
    pos_mean = positive_model_returns.mean()
    pos_std = positive_model_returns.std()

    # Calculate Drawdown
    df['equity_curve_model'] = (1 + model_returns).cumprod()
    df['equity_curve_benchmark'] = (1 + benchmark_returns).cumprod()
    df['peak'] = df['equity_curve_model'].cummax()
    df['drawdown'] = 100*(-1 +  (df['equity_curve_model']) / df['peak'])
    
    #Calculate returns
    df['returns_bh'] = (1 + benchmark_returns).cumprod()
    df['returns_strat'] = (1 + model_returns).cumprod()
    #
    if 1==2:
        # Plot Predictions
        plt.figure(figsize=(15, 8))
        ax1 = plt.subplot(3, 1, 1)
        pred_1 = df[df['pred'] == 1]
        ax1.scatter(pred_1.index, pred_1['pred'], marker='^', 
                c=np.where(pred_1['pred'] == pred_1['real'], 'b', 'r'), label='Predicción 1')
        pred_0 = df[df['pred'] == 0]
        ax1.scatter(pred_0.index, pred_0['pred'].shift(1), marker='v', 
                c=np.where(pred_0['pred'] == pred_0['real'], 'b', 'r'), label='Predicción 0')

        ax1.legend()
        ax1.set_title('Predictions')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Prediction')
        # Plot Returns
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(df.index, (1 + benchmark_returns).cumprod(), label="Equity Benchmark (B&H)")
        ax2.plot(df.index, (1 + model_returns).cumprod(), label="Equity Strat")
        ax2.set_title('Equity Curve')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulated Returns')
        ax2.legend()
        ax2.grid(True)

        # Plot Drawdown
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(df.index, df['drawdown'])
        ax3.fill_between(df.index, df['drawdown'], color='red', alpha=0.3)
        ax3.set_title('Drawdown')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown % ')

        plt.tight_layout()
        plt.show()

    return sharpe_ratio, sortino_ratio, df['drawdown'].min(), df['returns_strat'].iloc[-1], df['returns_strat'].max(), pos_max, pos_min, pos_mean, pos_std, neg_max, neg_min, neg_mean, neg_std


def test_models_super (model_name, data_sample, pca): 
    model = pickle.load(open('models/'+model_name, 'rb'))
    df = utils.read_parquet('data/'+data_sample)
    df = utils.add_features(df)
    if pca==1:
        pca = pickle.load(open('models/pca_model.pkl', 'rb'))
        standar_scaler = pickle.load(open('models/scale_model.pkl', 'rb'))
        filterd_cols = pickle.load(open('models/filtered_cols.pkl', 'rb'))
        df[filterd_cols] = standar_scaler.transform(df[filterd_cols])
        df_pca = pca.transform(df[filterd_cols])
        df_pca = pd.DataFrame(df_pca, columns=[f'PC{i+1}' for i in range(df_pca.shape[1])])

        #Concatenar la columna 'target' al DataFrame PCA
        df_pca = pd.concat([df_pca, df[['target', 'Close']].reset_index(drop=True)], axis=1)
        df = df_pca
        y_pred= model.predict(df.drop(['target', 'Close'], axis=1))
    else:
        y_pred = model.predict(df.drop('target', axis=1))

    df['pred'] = y_pred
    df["real"] = df['target']
    cm = confusion_matrix(df['real'], df['pred'])

    # Extraer Verdaderos Positivos (VP), Falsos Positivos (FP), Verdaderos Negativos (VN), y Falsos Negativos (FN)
    VP = cm[1, 1]
    FP = cm[0, 1]
    VN = cm[0, 0]
    FN = cm[1, 0]
    # Precisión
    precision = VP / (VP + FP)
    precision_neg = VN / (VN + FN)
    exactitud = (VP + VN) / (VP + VN + FP + FN)
    sensibilidad = VP / (VP + FN)
    especificidad = VN / (VN + FP)

    # F1-Score
    f1 = f1_score(df['real'], df['pred'])
    sharpe, sortino, mindraw, finalret, maxret, pos_max, pos_min, pos_mean, pos_std, neg_max, neg_min, neg_mean, neg_std = backtest(df)
    return_df = pd.DataFrame({
                'Model': [model_name],
                'Best params': [model.best_params_],
                'Precision': [precision],
                'Precision Neg': [precision_neg],
                'Exactitud': [exactitud],
                'Sensibilidad': [sensibilidad],
                'Especificidad': [especificidad],
                'F1 Score': [f1],
                'Confusion Matrix': [confusion_matrix],
                'Classification Report': [classification_report],
                'Sharpe Ratio': [sharpe],
                'Sortino Ratio': [sortino],
                'Max Drawdown': [mindraw],
                'Final Equity': [finalret],
                'Max Equity': [maxret],
                'Pos Max': [pos_max],
                'Pos Min': [pos_min],
                'Pos Mean': [pos_mean],
                'Pos Std': [pos_std],
                'Neg Max': [neg_max],
                'Neg Min': [neg_min],
                'Neg Mean': [neg_mean],
                'Neg Std': [neg_std]
                })
    return return_df


def test_models_deep (model_name, data_sample, pca): 
    if model_name == 'rnn_model.pth':
        model = utilsDL.RNN(150, 128, 2, 1)
    elif model_name == 'cnn_model.pth':
        model = utilsDL.ConvNet(150)
    else:
        model = utilsDL.Deep(150)
    we = torch.load('models/'+model_name)
    model.load_state_dict(we)
    model.eval() 
    df = utils.read_parquet('data/'+data_sample)
    df = utils.add_features(df)
    pca = pickle.load(open('models/pca_model.pkl', 'rb'))
    standar_scaler = pickle.load(open('models/scale_model.pkl', 'rb'))
    filterd_cols = pickle.load(open('models/filtered_cols.pkl', 'rb'))
    df[filterd_cols] = standar_scaler.transform(df[filterd_cols])
    
    df_input_tensor = torch.tensor(df[filterd_cols].values, dtype=torch.float32)
    if model_name == 'cnn_model.pth':
        df_input_tensor = df_input_tensor.unsqueeze(2)
    # Realizar la predicción
    with torch.no_grad():  # Desactiva el cálculo de gradientes
        y_pred_tensor = model(df_input_tensor)  # Realiza la predicción
    
    # Convertir las predicciones a un formato legible
    y_pred = y_pred_tensor.numpy()  # Convertir tensor a numpy array

    df['pred'] = np.where(y_pred > 0.6, 1, 0)
    df["real"] = df['target']
    cm = confusion_matrix(df['real'], df['pred'])

    # Extraer Verdaderos Positivos (VP), Falsos Positivos (FP), Verdaderos Negativos (VN), y Falsos Negativos (FN)
    VP = cm[1, 1]
    FP = cm[0, 1]
    VN = cm[0, 0]
    FN = cm[1, 0]
    # Precisión
    precision = VP / (VP + FP)
    precision_neg = VN / (VN + FN)
    exactitud = (VP + VN) / (VP + VN + FP + FN)
    sensibilidad = VP / (VP + FN)
    especificidad = VN / (VN + FP)

    # F1-Score
    f1 = f1_score(df['real'], df['pred'])
    sharpe, sortino, mindraw, finalret, maxret, pos_max, pos_min, pos_mean, pos_std, neg_max, neg_min, neg_mean, neg_std = backtest(df)
    return_df = pd.DataFrame({
                'Model': [model_name],
                'Best params': 'N/A',
                'Precision': [precision],
                'Precision Neg': [precision_neg],
                'Exactitud': [exactitud],
                'Sensibilidad': [sensibilidad],
                'Especificidad': [especificidad],
                'F1 Score': [f1],
                'Confusion Matrix': [confusion_matrix],
                'Classification Report': [classification_report],
                'Sharpe Ratio': [sharpe],
                'Sortino Ratio': [sortino],
                'Max Drawdown': [mindraw],
                'Final Equity': [finalret],
                'Max Equity': [maxret],
                'Pos Max': [pos_max],
                'Pos Min': [pos_min],
                'Pos Mean': [pos_mean],
                'Pos Std': [pos_std],
                'Neg Max': [neg_max],
                'Neg Min': [neg_min],
                'Neg Mean': [neg_mean],
                'Neg Std': [neg_std]
                })
    return return_df



def backtest_for_unsupervised(df, clases):
    df['pred'] = df['pred'].apply(lambda x: 1 if x in clases else 0)
    sharpe, sortino, mindraw, finalret, maxret, pos_max, pos_min, pos_mean, pos_std, neg_max, neg_min, neg_mean, neg_std = backtest(df)
    return finalret

def test_models_unsuper (model_name, data_sample, pca): 
    model = pickle.load(open('models/'+model_name, 'rb'))
    df = utils.read_parquet('data/'+data_sample)
    df = utils.add_features(df)
    if pca==1:
        pca = pickle.load(open('models/pca_model.pkl', 'rb'))
        standar_scaler = pickle.load(open('models/scale_model.pkl', 'rb'))
        filterd_cols = pickle.load(open('models/filtered_cols.pkl', 'rb'))
        df[filterd_cols] = standar_scaler.transform(df[filterd_cols])
        df_pca = pca.transform(df[filterd_cols])
        df_pca = pd.DataFrame(df_pca, columns=[f'PC{i+1}' for i in range(df_pca.shape[1])])

        #Concatenar la columna 'target' al DataFrame PCA
        df_pca = pd.concat([df_pca, df[['target', 'Close']].reset_index(drop=True)], axis=1)
        df = df_pca
        try:
            y_pred= model.best_estimator_.predict(df.drop(['target', 'Close'], axis=1))
        except:
            y_pred= model.best_estimator_.fit_predict(df.drop(['target', 'Close'], axis=1).values)
    else:
        pca = pickle.load(open('models/pca_model.pkl', 'rb'))
        standar_scaler = pickle.load(open('models/scale_model.pkl', 'rb'))
        filterd_cols = pickle.load(open('models/filtered_cols.pkl', 'rb'))
        df[filterd_cols] = standar_scaler.transform(df[filterd_cols])
        try:

            y_pred= model.best_estimator_.predict(df[filterd_cols])
        except:
            y_pred= model.best_estimator_.fit_predict(df[filterd_cols])
    df["real"] = df['target']
    df['pred'] = y_pred  
    print(df['pred'].value_counts())
    clases = []
    for i in range(0, y_pred.max()):
        c =[]
        c.append(i)
        ret = backtest_for_unsupervised(df, c)
        if ret > 1.0:
            clases.append(i)  
    print(clases)
    df['pred'] = df['pred'].apply(lambda x: 1 if x in clases else 0)
    cm = confusion_matrix(df['real'], df['pred'])

    # Extraer Verdaderos Positivos (VP), Falsos Positivos (FP), Verdaderos Negativos (VN), y Falsos Negativos (FN)
    VP = cm[1, 1]
    FP = cm[0, 1]
    VN = cm[0, 0]
    FN = cm[1, 0]
    # Precisión
    precision = VP / (VP + FP)
    precision_neg = VN / (VN + FN)
    exactitud = (VP + VN) / (VP + VN + FP + FN)
    sensibilidad = VP / (VP + FN)
    especificidad = VN / (VN + FP)

    # F1-Score
    f1 = f1_score(df['real'], df['pred'])
    sharpe, sortino, mindraw, finalret, maxret, pos_max, pos_min, pos_mean, pos_std, neg_max, neg_min, neg_mean, neg_std = backtest(df)
    return_df = pd.DataFrame({
                'Model': [model_name],
                'Best params': [model.best_params_],
                'Precision': [precision],
                'Precision Neg': [precision_neg],
                'Exactitud': [exactitud],
                'Sensibilidad': [sensibilidad],
                'Especificidad': [especificidad],
                'F1 Score': [f1],
                'Confusion Matrix': [confusion_matrix],
                'Classification Report': [classification_report],
                'Sharpe Ratio': [sharpe],
                'Sortino Ratio': [sortino],
                'Max Drawdown': [mindraw],
                'Final Equity': [finalret],
                'Max Equity': [maxret],
                'Pos Max': [pos_max],
                'Pos Min': [pos_min],
                'Pos Mean': [pos_mean],
                'Pos Std': [pos_std],
                'Neg Max': [neg_max],
                'Neg Min': [neg_min],
                'Neg Mean': [neg_mean],
                'Neg Std': [neg_std]
                })
    return return_df
       


# Definir una lista para almacenar los resultados de cada modelo
results = []
'''

'''
# Ejecutar cada modelo y almacenar los resultados en un DataFrame
models = [
('KMeans.pkl', 'U', 1), 
('DBSCAN.pkl', 'U', 1),
('GaussianMixture.pkl', 'U', 1),
('IsolationForest.pkl', 'U', 2),
('HiddenMarkovModel.pkl', 'U', 1),
('AdaBoostClassifier.pkl', 'S', 0),
('XGBClassifier.pkl', 'S', 0),
('CatBoostClassifier.pkl', 'S', 0),
('RandomForestClassifier.pkl', 'S', 0),
('KNeighborsClassifier.pkl', 'S', 0),
('RidgeClassifier.pkl', 'S', 1),
('SVC.pkl', 'S', 1),
('rnn_model.pth', 'D', 0),
('cnn_model.pth', 'D', 0),
('dnn_model.pth', 'D', 0),
]
columns = [
    'Model', 'Best params', 'Precision', 'Exactitud', 'Sensibilidad', 'Especificidad', 
    'F1 Score', 'Confusion Matrix', 'Classification Report', 
    'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Final Equity', 
    'Max Equity', 'Pos Max', 'Pos Min', 'Pos Mean', 'Pos Std', 
    'Neg Max', 'Neg Min', 'Neg Mean', 'Neg Std'
]

df_results = pd.DataFrame(columns=columns)

for model_name, type, pca in models:
    print(model_name)
    result_row = None
    if type == 'S':
        result_row = test_models_super(model_name, 'LTCUSDCHistData.parquet', pca)
    elif type == 'U':        
        result_row = test_models_unsuper(model_name, 'LTCUSDCHistData.parquet', pca)
    else:
        result_row = test_models_deep(model_name, 'LTCUSDCHistData.parquet', 1)
    if result_row is not None:
        df_results  = pd.concat([df_results, result_row])
    else:
        print(f'Error en el modelo {model_name}')
print(df_results)




