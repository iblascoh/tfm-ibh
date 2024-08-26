
from matplotlib import  pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
import ta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import pickle
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def backtest(df):

    # Calculate Sortino Ratio
    df['pred'] = df['pred'].shift(1)
    df['real'] = df['real'].shift(1)
    df.dropna(inplace=True)
    model_returns = df['pred'] * df['Close'].pct_change(1)
    negative_model_returns = model_returns[model_returns < 0]
    downside_std = negative_model_returns.std()
    sortino_ratio = 100*np.sqrt(len(df))*(model_returns.mean() - 0) / downside_std

    # Calculate Alpha and Beta
    benchmark_returns = df['Close'].pct_change(1)
    benchmark_returns = benchmark_returns.fillna(0)
    
    # Calculate Alpha and Beta
    covariance = np.cov(model_returns, benchmark_returns)[0, 1]
    beta = covariance / benchmark_returns.var()
    alpha = np.mean(model_returns) - beta * np.mean(benchmark_returns)

    # Calculate Sharpe Ratio
    risk_free_rate = 0.03  # Assuming no risk-free rate
    excess_returns = model_returns - risk_free_rate
    sharpe_ratio = 100*np.sqrt(len(df))* excess_returns.mean() / excess_returns.std()

    # Calculate Drawdown
    df['equity_curve_model'] = (1 + model_returns).cumprod()
    df['equity_curve_benchmark'] = (1 + benchmark_returns).cumprod()
    df['peak'] = df['equity_curve_model'].cummax()
    df['drawdown'] = 100*(-1 +  (df['equity_curve_model']) / df['peak'])

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

    print(f"Max Drawdown: {df['drawdown'].min()}")
    print(f"Max Drawdown Duration: {df['drawdown'].idxmax()} - {df['drawdown'].idxmin()}")
    print(f"Sortino Ratio: {sortino_ratio}")
    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Alpha: {alpha}")
    print(f"Beta: {beta}")

    df['equity_curve_model']
    df['equity_curve_benchmark'] 
    return df['equity_curve_model'].iloc[-1], df['equity_curve_benchmark'].iloc[-1]

def read_parquet(file):
    df = pd.read_parquet(file)  # Leer el archivo parquet en un dataframe
    # Convertir las columnas a los tipos de datos correctos
    df[["Open", "High", "Low", "Close", "Volume"]] = df[["open", "high", "low", "close", "volume"]] .astype(float)  # Convertir todas las columnas a tipo float
    df['openTime'] = pd.to_datetime(df['openTime'], format='%d-%m-%Y %H:%M:%S.%f')
    df.index = pd.DatetimeIndex(df['openTime'])
    df.shape
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df

def add_features(df):
    df['returns'] = df["Close"].pct_change(1)
    # Día de la semana y hora del día
    df['Day'] = df.index.dayofweek
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    
    # Indicadores técnicos
    df['Sma_15'] =ta.trend.sma_indicator(df['Close'], window=15, fillna=True)
    df['Sma_10'] =ta.trend.sma_indicator(df['Close'], window=10, fillna=True)
    df['Ema_15'] =ta.trend.ema_indicator(df['Close'], window=15, fillna=True)
    df['Ema_10'] =ta.trend.ema_indicator(df['Close'], window=10, fillna=True)
    df['Rsi_15'] = ta.momentum.rsi(df['Close'], window=15, fillna=True)
    df["Sti_14"] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14, fillna=True)
    df['Macd'] = ta.trend.macd_diff(df['Close'], window_slow=14, window_fast=7, window_sign=4, fillna=True)
    df['Bollinger_hband'] = ta.volatility.bollinger_hband(df['Close'], window=15, fillna=True)
    df['Bollinger_lband'] = ta.volatility.bollinger_lband(df['Close'], window=15, fillna=True)
    df['Atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=15, fillna=True)
    df['Obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'], fillna=True)
    df['Adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=15, fillna=True)
    df['Cci'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=15, fillna=True)
    df['Dpo'] = ta.trend.dpo(df['Close'], window=15, fillna=True)
    df['Trix'] = ta.trend.trix(df['Close'], window=15, fillna=True)


    df['Buy_signal1'] = np.where((df['Rsi_15'] < 30) & (df['Close'] < df['Bollinger_lband']), 1, 0)
    df['Sell_signal1'] = np.where((df['Rsi_15'] > 70) & (df['Close'] > df['Bollinger_hband']), 1, 0)
    df['Composite_signal'] = df['Sma_15'] + df['Ema_15'] + df['Rsi_15'] + df['Macd'].rolling(window=3).mean()
    df['Buy_signal2'] = np.where(df['Composite_signal'] > 0, 1, 0)
    df['Sell_signal2'] = np.where(df['Composite_signal'] < 0, 1, 0)
    df['Buy_signal3'] = np.where(df['Composite_signal'] > 0, 1, 0)
    df['Sell_signal3'] = np.where(df['Composite_signal'] < 0, 1, 0)
    df['TRIX_signal'] = np.where(df['Trix'] > df['Trix'].rolling(window=10).mean(), 1, 0)
    df['DPO_signal'] = np.where(df['Dpo'] > 0, 1, 0)
    df['CCI_signal'] = np.where(df['Cci'] > 100, 0, np.where(df['Cci'] < -100, 1, 0))
    df['ADX_signal'] = np.where(df['Adx'] > 25, 1, 0)
    


    # Indicadores sobre precio
    df['Price_change'] = df['Close'].pct_change(1)
    #df['Price_change_mean'] = df['Price_change'].rolling(window=15).mean()
    #df['Price_change_std'] = df['Price_change'].rolling(window=15).std()
    #df['Volume_change'] = df['Volume'].pct_change(1)
    #df['Volume_change_mean'] = df['Volume_change'].rolling(window=15).mean()
    #df['Volume_change_std'] = df['Volume_change'].rolling(window=15).std()
    df['Mom'] = df['Close'].pct_change(periods=2)
    #df['Mom_mean'] = df['Mom'].rolling(window=15).mean()
    #df['Mom_std'] = df['Mom'].rolling(window=15).std()


    for col in df.columns:
        for lag in range(1, 5):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    df['target'] = np.where(df['returns'].shift(-1)>0, 1, 0)
    df.dropna(inplace=True)
    return df

def train_test_split(df, size=0.2):
    test_size = int(len(df) * size)
    X_train = df.drop(['target'], axis=1)[:-test_size]
    X_test = df.drop(['target'], axis=1)[-test_size:]
    y_train = df['target'][:-test_size]
    y_test = df['target'][-test_size:]
    #print(X_train.columns.tolist())
    return X_train, X_test, y_train, y_test

def scale_down(X_train, X_test, option = 0):

    if option == 0:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    print("Scaler done")
    return X_train_sc, X_test_sc, scaler

def reduce_dim (X_train, X_test):
    pca = PCA (n_components =10 )
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("Variance explained by PCA: ", pca.explained_variance_ratio_)
    print("PCA done")
    return X_train_pca, X_test_pca, pca

def classification_score(X_test, y_test):


    confusion = confusion_matrix(y_test, X_test["pred"])
    print("Confusion Matrix:")
    print(confusion)
    f1 = f1_score(y_test, X_test["pred"])
    accuracy = accuracy_score(y_test, X_test["pred"])
    recall = recall_score(y_test, X_test["pred"])
    precision = precision_score(y_test, X_test["pred"])

    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")

def filter_features (X, y, num_features, columns, prt=False):
    model = Lasso(alpha=0.0001, max_iter=10000000)
    model.fit(X, y)
    feature_names= columns
    # Obtener coeficientes
    coef = np.abs(model.coef_)

    # Crear un DataFrame para visualizar
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': coef
    }).sort_values(by='Importance', ascending=False)
    if prt ==True: 
        print("Caracteristicas mas relevantes", feature_importance_df['Feature'].head(num_features).tolist())
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title('Feature Importance')
        plt.show()
    
    # Filtrar las num_features características más relevantes
    selected_features = feature_importance_df['Feature'].head(num_features).tolist()
    
    return selected_features

def silhouette_scorer(estimator, X):
    cluster_labels = estimator.fit_predict(X)
    if len(set(cluster_labels)) > 1:
        return silhouette_score(X, cluster_labels)
    else:
        return -1
    
def get_best_model_with_tuning_supervised(X_train, y_train, X_test, y_test, X_train_pca, X_test_pca, models):
    best_model = None
    best_score = 0
    
    # Define the parameter grids
    param_grids = {
        'SVC': {
            'C': [ 0.0001, 0.00001],
            'kernel': ['linear'], 
        },
        'AdaBoostClassifier': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.001, 0.01, 0.1, 1]
        },
        'XGBClassifier': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 10]
        },
        'CatBoostClassifier': {
            'iterations': [50, 100],
            'learning_rate': [0.01, 0.1],
            'depth': [6, 10]
        },
        'RidgeClassifier': {
            'alpha': [0.1, 1, 10]
        },
        'LinearDiscriminantAnalysis': {
            'solver': ['svd', 'lsqr', 'eigen']
        },
        'RandomForestClassifier': {
            'n_estimators': [50, 100, 200],
            'max_depth': [ 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'KNeighborsClassifier': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    }
    modelos = []
    names = []
    # GridSearchCV for each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        param_grid = param_grids.get(model_name, {})
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=False) 
        
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_strategy, scoring='precision', verbose=3)
        if (model_name == 'SVC' or model_name == 'RidgeClassifier'):
            grid_search.fit(X_train_pca, y_train)
            
            print(f"Best Model for {model_name}: {grid_search.best_params_}")
            print(f"Best Score for {model_name}: {grid_search.best_score_}")
            print(f"Best Parameters for {model_name}: {grid_search.best_estimator_}")
            print(confusion_matrix(y_test, grid_search.predict(X_test_pca)))
        else: 
            grid_search.fit(X_train, y_train)
            print(f"Best Model for {model_name}: {grid_search.best_params_}")
            print(f"Best Score for {model_name}: {grid_search.best_score_}")
            print(f"Best Parameters for {model_name}: {grid_search.best_estimator_}")
            print(confusion_matrix(y_test, grid_search.predict(X_test)))

        modelos.append(grid_search)
        names.append(model_name)
    return modelos, names

def get_best_model_with_tuning_unsupervised(X_train_pca, X_train_sc, models):
    best_model = None
    best_score = 0
    
    # Define the parameter grids
    param_grids = {
        'KMeans': {
            'n_clusters': [3, 5, 7],
            'init': ['k-means++', 'random']
        },
        'DBSCAN': {
            'eps': [ 2, 3, 4],
            'min_samples': [ 15, 100, 150]
        },
        'GaussianMixture': {
            'n_components': [4, 8, 12],
            'covariance_type': ['full', 'tied', 'spherical']
        },
        'IsolationForest': {
            'n_estimators': [50, 100, 200],
            'contamination': [0.1, 0.2, 0.3]
        },
        'HiddenMarkovModel': {
            'n_iter': [1000, 1500],
            'covariance_type': ['full', 'tied', 'diag'],
            'n_components': [4, 8, 12]
        }
    }


    modelos = []
    names = []
    # GridSearchCV for each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        param_grid = param_grids.get(model_name, {})
        grid_search = RandomizedSearchCV(estimator=model, cv = 2, param_distributions=param_grid, scoring=silhouette_scorer, verbose=3)
        if  model_name == 'IsolationForest':
            grid_search.fit(X_train_sc)
        else:
            grid_search.fit(X_train_pca)
        modelos.append(grid_search)
        names.append(model_name)
    return modelos, names

def train_save_supervised_models(files, models):
        df= pd.DataFrame()
        for (file) in files:
            df_aux = read_parquet('data/'+file) # type: ignore
            df_aux = add_features(df_aux)
            print(df_aux.isna().sum())
            print(df_aux.shape)
            df = pd.concat([df, df_aux], axis=0) # type: ignore

        print("Datasets cargados")
        print(df.shape)
        #df = add_features(df) # type: ignore  
        X_train, X_test, y_train, y_test = train_test_split(df) # type: ignore 
        X_train_sc, X_test_sc, scale_model = scale_down(X_train, X_test, 0) # type: ignore
        cols = filter_features (X_train_sc, y_train, 150, columns = X_train.columns) # type: ignore
        X_train_sc, X_test_sc, scale_model = scale_down(X_train.filter(items = cols), X_test.filter(items = cols), 0) # type: ignore
        X_train_pca, X_test_pca, pca_model = reduce_dim(X_train_sc, X_test_sc) # type: ignore

        modelos, names = get_best_model_with_tuning_supervised(X_train, y_train, X_test, y_test, X_train_pca, X_test_pca, models)



        pickle.dump(cols, open(f"models/filtered_cols.pkl", "wb"))
        pickle.dump(scale_model, open(f"models/scale_model.pkl", "wb"))
        pickle.dump(pca_model, open(f"models/pca_model.pkl", "wb"))
        for model, name in zip(modelos, names):
            pickle.dump(model, open(f"models/{name}.pkl", "wb"))

def train_save_unsupervised_models(files, models):
        df= pd.DataFrame()
        for (file) in files:
            df_aux = read_parquet('data/'+file) # type: ignore
            print(df_aux.shape)
            df_aux = add_features(df_aux)
            print(df_aux.isna().sum())
            print(df_aux.shape)
            df = pd.concat([df, df_aux], axis=0) # type: ignore 
        X_train, X_test, y_train, y_test = train_test_split(df) # type: ignore 
        X_train_sc, X_test_sc, scale_model = scale_down(X_train, X_test, 0) # type: ignore
        cols = filter_features (X_train_sc, y_train, 150, columns = X_train.columns) # type: ignore
        X_train_sc, X_test_sc, scale_model = scale_down(X_train.filter(items = cols), X_test.filter(items = cols), 0) # type: ignore
        X_train_pca, X_test_pca, pca_model = reduce_dim(X_train_sc, X_test_sc) # type: ignore

        modelos, names = get_best_model_with_tuning_unsupervised(X_train_pca, X_train_sc, models)


        pickle.dump(cols, open(f"models/filtered_cols.pkl", "wb"))
        pickle.dump(scale_model, open(f"models/scale_model.pkl", "wb"))
        pickle.dump(pca_model, open(f"models/pca_model.pkl", "wb"))
        for model, name in zip(modelos, names):
            pickle.dump(model, open(f"models/{name}.pkl", "wb"))
