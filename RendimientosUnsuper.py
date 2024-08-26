from matplotlib import pyplot as plt
import pandas as pd
import pickle
import SupervisedLearning as utils
import numpy as np


model = pickle.load(open('models/HiddenMarkovModel.pkl', 'rb'))
df = utils.read_parquet('data/'+'LTCUSDCHistData.parquet')
df = utils.add_features(df)
pca = pickle.load(open('models/pca_model.pkl', 'rb'))
standar_scaler = pickle.load(open('models/scale_model.pkl', 'rb'))
filterd_cols = pickle.load(open('models/filtered_cols.pkl', 'rb'))
df[filterd_cols] = standar_scaler.transform(df[filterd_cols])
df_pca = pca.transform(df[filterd_cols])
df_pca = pd.DataFrame(df_pca, columns=[f'PC{i+1}' for i in range(df_pca.shape[1])])

#Concatenar la columna 'target' al DataFrame PCA
df_pca = pd.concat([df_pca, df[['target', 'Close']].reset_index(drop=True)], axis=1)
df = df_pca
y_pred= model.best_estimator_.predict(df.drop(['target', 'Close'], axis=1))
df['pred'] = y_pred

df['returns'] = df['Close'].pct_change()
df['bh_returns'] = (1+df['returns']).cumprod()
for i in range(10):
    df[f'cluster_{i}'] = (1+(df['returns'] * np.where(df['pred'].shift(1)==i, 1, 0))).cumprod()
print(df.head())
plt.plot(df['bh_returns'], label='bh_returns', color='black')
for i in range(10):
    plt.plot(df[f'cluster_{i}'], label=f'strat_{i}')
plt.title('HMM rets by cluster')
plt.legend(loc='upper left')
plt.show()




