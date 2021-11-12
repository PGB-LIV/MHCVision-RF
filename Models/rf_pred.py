import pandas as pd
import numpy as np
import pickle
import Models.peptide_label_feat


def load_model():
    model = pickle.load(open('./Models/Immune_RF_Model.sav', 'rb'))
    return model


def logistic_regression(predicted_prob):
    n = 0.048
    a = 5.87
    b = 2.892
    immune_prob = []
    for i in range(len(predicted_prob)):
        x = predicted_prob[i]
        y = n + (1 / (1 + np.exp((-1*a) * x + b)))
        immune_prob.append(y)
    return immune_prob


def make_prediction(df_matrix):
    peptides = df_matrix.loc[:,'peptides']
    predicted_data = np.nan_to_num(df_matrix.iloc[:,1:])  # exclude peptide column
    clf = load_model()
    y_prob = clf.predict_proba(predicted_data)
    predicted_prob = y_prob[:, 1]
    # convert predicted_prob to immune_prob via logistic regression transformation
    output_name = 'Immune_pred.csv'
    immune_prop = logistic_regression(predicted_prob)
    predicted_result = pd.DataFrame()
    predicted_result['Peptide'] = peptides
    predicted_result['Immunogenic probability'] = immune_prop
    predicted_result.to_csv(output_name, index=False)
    return


def immune_pred():
    df_matrix = Models.peptide_label_feat.generate_matrix('peptide.txt')
    make_prediction(df_matrix)
    return


