from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import AdaBoostClassifier, IsolationForest, RandomForestClassifier
from xgboost import XGBClassifier
from hmmlearn import hmm
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import  RidgeClassifier
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score
import MachineLearning as utils
import DeepLearning as utilsDL
import pickle

models_supervised = {
        'SVC': SVC(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'XGBClassifier': XGBClassifier(),
        'CatBoostClassifier': CatBoostClassifier(silent=True),
        'RidgeClassifier': RidgeClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'KNeighborsClassifier': KNeighborsClassifier()
}
        

models_unsupervised = {
        'KMeans': KMeans(),
        'DBSCAN': DBSCAN(),
        'GaussianMixture': GaussianMixture(),
        'IsolationForest': IsolationForest(),
        'HiddenMarkovModel':  hmm.GaussianHMM()
}




list_of_files = [
        'BTCUSDCHistData.parquet',
        'ETHUSDCHistData.parquet',]
def train_ml(list_of_files):

        m_super = models_supervised
        m_unsuper = models_unsupervised
        utils.train_save_supervised_models(list_of_files, m_super)
        utils.train_save_unsupervised_models(list_of_files, m_unsuper)


def train_dl(list_of_files):
        train_loader, X_test, y_test = utilsDL.load_data(list_of_files, 65)
#        dnn_model, dnn_loss = utilsDL.train_dnn_model(train_loader, X_test, y_test, 0.00001, 1)
#        rnn_model, rnn_loss = utilsDL.train_rnn_model(train_loader, X_test, y_test, 0.00001, 1)
        cnn_model, cnn_loss = utilsDL.train_convnet_model(train_loader, X_test, y_test, 0.0001, 1)
#train_ml(list_of_files)
train_dl(list_of_files)










