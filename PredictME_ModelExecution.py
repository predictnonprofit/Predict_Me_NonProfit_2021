import numpy as np
import pandas as pd
import random

import re
import json
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from datetime import date
from fpdf import FPDF
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
import sys
import warnings
import time
import os
import ast
import glob
import operator
import math


import locale
locale.setlocale(locale.LC_ALL, 'en_US')
font_style = 'Arial'


class CustomPDF(FPDF):
    def footer(self):
        self.set_y(-10)
        self.set_font(font_style, 'I', 8)

        # Add a page number
        page = 'Page ' + str(self.page_no())
        self.cell(0, 10, page, 0, 0, 'C')


warnings.filterwarnings("ignore")
pdf = CustomPDF()
pdf.set_font(font_style, size=10)
pdf.add_page()
image_index = 0


def convert_number_format(d):
    return locale.format("%d", d, grouping=True)

# Remove rows containing null values in all columns
def remove_rows_containg_all_null_values(df):
    idx = df.index[~df.isnull().all(1)]
    df = df.ix[idx]
    return df


# Read input donation file
def read_input_file(file_path):
    file_name = file_path.split('/')[-1]
    extension = file_name.split(".")[-1]
    if extension == "csv":
        return pd.read_csv(file_path, encoding="ISO-8859-1")
    elif (extension == "xlsx") | (extension == "xls"):
        return pd.read_excel(file_path, encoding="ISO-8859-1")
    else:
        print("{} file format is not supported".format(extension))


# Identify donation columns for file stored in DataStore
def identify_years_columns(file_name):
    mapping_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "column_name_mapping.json"))
    with open(mapping_file_path) as jsonfile:
        data = json.load(jsonfile)
    for k, v in data.items():
        if k in file_name:
            return v
    return "[]"


# Identify text columns
def identify_info_columns(df, donation_columns):
    column_names = df.columns
    return [col for col in column_names if col not in donation_columns]


# Remove column contains 80% unique values and more than 50% null values
def remove_columns_unique_values(df, column_names):
    final_col_ = []
    number_of_sample = df.shape[0]
    for col in column_names:
        # print("Column: {}, Null count: {}".format(col,
        # df[col].isnull().sum()))
        if df[col].isnull().sum() <= number_of_sample / 2:
            final_col_.append(col)
    final_col = []
    for col in final_col_:
        # print("Column: {}, Unique count: {}".format(col,
        # df[col].unique().shape[0]))
        if (df[col].unique().shape[0] <= number_of_sample * 0.8) and (df[col].unique().shape[0] != 1):
            final_col.append(col)
    final_df = df[final_col]
    final_df.fillna("", inplace=True)
    return final_df


# Identify unique categorical columns
def identify_categorical_columns(df, column_names):
    cat_col_ = []
    for col in column_names:
        if df[col].unique().shape[0] <= 5:
            cat_col_.append(col)
    return cat_col_


# Regex function to clean input text
def text_processing(text):
    pre_text = []
    for x in text:
        x = re.sub('[^a-zA-Z.\d\s]', '', x)
        x = re.sub(' +', ' ', x)
        x = str(x.strip()).lower()
        replace_strings = {"null": "", "nul": "", "nulls": ""}
        for old_str, new_str in replace_strings.items():
            x = x.replace(old_str, new_str)
        pre_text.append(x)
    return pre_text


# Convert text to numeric values using Tf-IDF
def feature_extraction(df_info):
    df_info = df_info.astype(str)
    df_info['comb_text'] = df_info.apply(lambda x: ' '.join(x), axis=1)
    processed_text = text_processing(list(df_info['comb_text']))
    unique_features = len(Counter([" ".join(x for x in processed_text)][0].split()).keys())
    feature_count = int(0.5 * unique_features)

    if feature_count >= len(df_info) * 0.7:
        feature_count = math.ceil(len(df_info) * 0.6)

    elif feature_count <= 1000:
        feature_count = 1000
    elif feature_count >= 3000:
        feature_count = 3000
    else:
        feature_count = int(0.5 * unique_features)
    print("unique features {} and feature count {}".format(unique_features, feature_count))
    vectorizer = TfidfVectorizer(max_features=feature_count)
    X = vectorizer.fit_transform(processed_text)
    tfidf_matrix = X.todense()
    feature_names = vectorizer.get_feature_names()
    return processed_text, tfidf_matrix, feature_names, df_info, vectorizer


# Clean donation columns by keeping only digits
def clean_donation(donation):
    donation = ''.join(c for c in donation if (c.isdigit()) | (c == "."))
    if donation == "":
        return "0"
    else:
        return donation


# Identify target value for each record
def process_donation_columns(df, donation_columns, no_donation_columns, skewed_target_value):
    if no_donation_columns:
        donation_columns = ast.literal_eval(donation_columns)
    elif skewed_target_value:
        donation_columns = ast.literal_eval(donation_columns)

    donation_columns = df[donation_columns].fillna("0")
    donation_columns = donation_columns.astype(str)

    for col in donation_columns.columns:
        donation_columns[col] = donation_columns[col].apply(lambda x: clean_donation(x))
        donation_columns[col] = donation_columns[col].astype(float)

    donation_columns = donation_columns.astype(int)

    no_of_col = donation_columns.shape[1]

    def identify_target(x):
        non_zero_col = 0
        for col in donation_columns.columns:
            if x[col] > 0:
                non_zero_col += 1
        return non_zero_col

    donation_columns['donation_non_zero'] = donation_columns.apply(lambda x: identify_target(x), axis=1)
    col_threshold = int(no_of_col / 2.)
    donation_columns['target'] = donation_columns['donation_non_zero'].apply(lambda x: 1 if x > col_threshold else 0)
    del donation_columns['donation_non_zero']
    return donation_columns

# Measuring Skewness of the file and check to see if donation columns exist
def check_skew_donation_cols(donation_columns_df, donation_columns):

    no_donations_columns = False
    skewed_target_value = False

    postive_class = donation_columns_df[donation_columns_df['target'] == 1].shape[0]
    negative_class = donation_columns_df[donation_columns_df['target'] == 0].shape[0]

    if(postive_class == 0 or negative_class == 0):
        no_donations_columns = True


    # check for donation columns
    if ((len(donation_columns) == 0) | (donation_columns == "[]")):
        no_donations_columns = True
        print("No donation columns present for the file")
    else:
        print("donation columns {}".format(len(donation_columns)))

    # check for skewness
    if ((postive_class <= math.ceil((donation_columns_df.shape[0]) * 0.02)) | (negative_class <= math.ceil((donation_columns_df.shape[0]) * 0.02))):
        skewed_target_value = True
        print("positive class {} and negative class {}".format(postive_class, negative_class))
                              
    return no_donations_columns, skewed_target_value


# Generate correlation plot for donation columns
def generate_correlation(donation_columns, no_donation_columns, skewed_target_value):
    if no_donation_columns:
        pdf.set_font(font_style, 'BU', size=10)
        pdf.multi_cell(h=5.0, w=0, txt="# Correlation Plot")
        pdf.set_font(font_style, size=10)
        pdf.ln(1)
        pdf.multi_cell(h=5.0, w=0, txt="Correlation Plot: Please note that Correlation is calculated based on the similar "
                                       "donor datasets stored in Predict Me's server. These datasets are used to find "
                                       "common donor attributes to maximize the model performance.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="NOTE: The uploaded donor file is missing donation information (amount) required"
                                       " for plotting a Correlation Matrix.")
        pdf.ln(2)
    elif skewed_target_value:
        pdf.set_font(font_style, 'BU', size=10)
        pdf.multi_cell(h=5.0, w=0, txt="# Correlation Plot")
        pdf.set_font(font_style, size=10)
        pdf.ln(1)
        pdf.multi_cell(h=5.0, w=0, txt="Correlation Plot: The uploaded donor file has an imbalanced dataset. More than 98% "
                                       "of your sample belongs to one class (0 or 1 Target Value) that make up a large "
                                       "proportion of the data.")
        pdf.ln(0.5)

        pdf.multi_cell(h=5.0, w=0, txt="Please note that Correlation is calculated based on the similar donor datasets "
                                       "stored in Predict Me's server. These datasets are used to avoid imbalanced data"
                                       " issues and find common donor attributes to maximize the model performance.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="NOTE: The uploaded donor file is missing text values (attributes) required for "
                                       "plotting a Correlation.")
        pdf.ln(2)
    else:
        pdf.ln(2)
        sn.set(font_scale=2)
        fig, ax = plt.subplots(figsize=(25, 25))
        ax = sn.heatmap(donation_columns.corr().round(2).replace(-0, 0), annot=True)
        plt.title('Correlation Plot', fontsize=45)
        global image_index
        plots_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Plots"))
        plt.savefig("{}/temp_{}.png".format(plots_path, image_index))
        pdf.image("{}/temp_{}.png".format(plots_path, image_index), w=175, h=150)
        image_index += 1


# Calculate sum of feature weights
def get_feature_weights(feature_list, feature_dict):
    sum_ = 0
    for f in feature_list:
        sum_ += feature_dict.get(f.lower(), 0)
    return sum_


# Plot feature importance for each column
def calculate_feature_importance(df_info, feature_names, feature_value, no_donation_columns, skewed_target_value, skewed_target_value_similar, is_similar_file):

    feature_dict = {i: abs(j) for i, j in zip(feature_names, feature_value)}
    info_columns = list(df_info.columns)
    info_columns.remove('comb_text')
    
    feature_dict_ = {}
    for col in info_columns:
        feat = [x.split() for x in df_info[col]]
        flat_list = [item for sublist in feat for item in sublist]
        feature_dict_[col] = get_feature_weights(text_processing(flat_list), feature_dict)
            
        
    total_sum = sum(feature_dict_.values())
    
    feature_imp = list(feature_dict_.values()) / (total_sum / 100)
    feature_columns = list(feature_dict_.keys())

    pdf.set_font(font_style, 'BU', size=10)
    pdf.multi_cell(h=5.0, w=0, txt="D (d). Feature Importance Plot")

    sn.set(font_scale=2)
    sorted_idx = np.argsort(feature_imp)
    pos = np.arange(sorted_idx.shape[0]) + .5
    if (no_donation_columns or is_similar_file == True):

        pdf.set_font(font_style, size=10)
        pdf.ln(1)
        pdf.multi_cell(h=5.0, w=0, txt="Note: Feature Importance is based on the testing set of the similar donor file stored in Predict Me's server."
                                       "No categorical features processed from the input file for plotting a Feature Importance.")
        pdf.ln(2)
    
    else:
        featfig = plt.figure(figsize=(10, 6))
        featax = featfig.add_subplot(1, 1, 1)
        featax.barh(pos, sorted(feature_imp), align='center')
        featax.set_yticks(pos)
        featax.set_yticklabels(np.array(feature_columns)[sorted_idx], fontsize=12)
        featax.set_xlabel('% Relative Feature Importance', fontsize=16)
        # featax.set_xticklabels(fontsize=12)
        plt.tight_layout()
        plt.title('Feature Importance Plot', fontsize=16)
        global image_index
        plots_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Plots"))
        plt.savefig("{}/temp_{}.png".format(plots_path, image_index))
        pdf.image("{}/temp_{}.png".format(plots_path, image_index), w=170, h=102)
        image_index += 1
        pdf.ln(2)


# Generate classification report
def add_classification_report_table(y_test, y_pred, no_donation_columns, skewed_target_value, is_similar_file):
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df['f1-score'] = report_df['f1-score'].apply(lambda x: str(round(x, 2)))
    report_df['precision'] = report_df['precision'].apply(lambda x: str(round(x, 2)))
    report_df['recall'] = report_df['recall'].apply(lambda x: str(round(x, 2)))
    del report_df['support']
    #report_df['support'] = report_df['support'].apply(lambda x:
    #str(convert_number_format(int(x))))
    report_df = report_df.rename(columns={"f1-score": "F1-score",
                                          "precision": "Precision",
                                          "recall": "Recall"})
                                          #"support": "Support"})
    report_df.reset_index(inplace=True)
    report_df['index'] = report_df['index'].replace({"0": "Non-donor class", "1": "Donor class"})
                                                     #"macro avg": "Macro avg",
                                                     #"weighted avg": "Weighted
                                                     #avg"
    report_df = report_df.rename(columns={"index": "Class"})
    report_df = pd.DataFrame(np.vstack([report_df.columns, report_df]))
    report_df = report_df.values.tolist()
    del report_df[3:6]
    spacing = 1.25
    col_width = pdf.w / 6
    row_height = pdf.font_size
    # if donation columns do not exis, the classification report is based on
    # the similar file not the real input file
    if(no_donation_columns or is_similar_file == True):
        pdf.multi_cell(h=5.0, w=0, align ='L', txt="Note: Classification Repot Table is based on the testing set of the similar donor file stored in Predict Me's server.")
        pdf.ln(4)

    for row in report_df:
        for item in row:
            pdf.cell(col_width, row_height * spacing, txt=item, border=1, align="C")
        pdf.ln(row_height * spacing)


# Plot confusion matrix
def print_confusion_matrix_classification_report(y_test, y_pred, no_donation_columns, skewed_target_value, is_similar_file):
    df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(2), range(2))
    plt.figure(figsize=(15, 10))
    sn.set(font_scale=2.5) # for label size
    sn.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 30}) # font size
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tick_params(axis="both", which="both", labelsize="large")
    global image_index

    pdf.set_font(font_style, 'BU', size=10)
    pdf.multi_cell(h=7.5, w=0, txt="D (a). Confusion Matrix Plot")

    pdf.set_font(font_style, size=10)
    # pdf.multi_cell(h=5.0, w=0, txt="# Confusion Matrix Plot")
    if (no_donation_columns or is_similar_file == True):
        pdf.multi_cell(h=5.0, w=0, align ='L', txt="Note: Confusion Matrix Plot is based on the testing set of the similar donor file stored in Predict Me's server.")

    pdf.ln(1)
    pdf.set_font(font_style, size=10)
    plt.title('Confusion Matrix Plot', fontsize=36)
    plots_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Plots"))
    plt.savefig("{}/temp_{}.png".format(plots_path, image_index))
    pdf.image("{}/temp_{}.png".format(plots_path, image_index), w=112, h=75)
    image_index += 1

    pdf.ln(0.5)
    pdf.set_font(font_style, size=10)
    

    pdf.multi_cell(h=5.0, w=0, align ='L', txt="Based on the confusion matrix, a total of {} ({} + {}) samples belong to class donor and {} ({} + {}) samples belong to class non-donor. "
                                   "The model correctly predicted {} samples as class donor and {} samples as class non-donor. "
                                   "The model misclassified {} class donor as class non-donor and {} class non-donor as class donor."
                                   .format(convert_number_format((df_cm.values[0:2][1][0] + df_cm.values[0:2][1][1])),
                                           convert_number_format(df_cm.values[0:2][1][0]),
                                           convert_number_format(df_cm.values[0:2][1][1]),
									       convert_number_format((df_cm.values[0:2][0][0] + df_cm.values[0:2][0][1])),
                                           convert_number_format(df_cm.values[0:2][0][0]),
                                           convert_number_format(df_cm.values[0:2][0][1]),
										   convert_number_format(df_cm.values[0:2][1][1]),
										   convert_number_format(df_cm.values[0:2][0][0]),
										   convert_number_format(df_cm.values[0:2][1][0]),
										   convert_number_format(df_cm.values[0:2][0][1])))
    
    pdf.ln(0.5)
    pdf.ln(4)
    pdf.set_font(font_style, 'BU', size=10)
    pdf.multi_cell(h=5.0, w=0, txt="D (b). Classification Report Table")
    pdf.set_font(font_style, size=10)
    pdf.ln(4)
    add_classification_report_table(y_test, y_pred, no_donation_columns, skewed_target_value, is_similar_file)
    pdf.ln(4)


# Calculates false postitive and true positive rate
def calculate_fpr_tpr(model, y_test, y_pred, X_test, y_prob):
    try:
        if model != None and X_test != None:
            fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        else:
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
    except:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
    return fpr, tpr, auc


# Plot ROC curve
def plot_roc_curve(roc_fpr, roc_tpr, roc_auc, top_5_models, no_donation_columns, skewed_target_value, is_similar_file):
    pdf.set_font(font_style, 'BU', size=10)
    pdf.multi_cell(h=5.0, w=0, txt="D (c). ROC Curve")
    pdf.set_font(font_style, size=10)
    pdf.ln(2)
    pdf.multi_cell(h=5.0, w=0, txt="A model with high accuracy is represented by a line that travels from the bottom left of the plot to the top left "
                                   "and then across the top to the top right and has Area Under Curve (AUC) as 1. A model with less accuracy is represented by a "
                                  "diagonal line from the bottom left of the plot to the top right and has an AUC of 0.5. The best model has AUC close to 1.")
    
    pdf.ln(1)
    global image_index
    if (no_donation_columns or is_similar_file == True):
        pdf.multi_cell(h=5.0, w=0, txt="Note: ROC is based on the testing set of the similar donor file stored in Predict Me's server.")
        pdf.ln(1)

        pdf.ln(2)
        plt.figure(figsize=(15, 10))
        sn.set(font_scale=2)
        for model_name in top_5_models:
            fpr = roc_fpr.get(model_name)
            tpr = roc_tpr.get(model_name)
            auc = roc_auc.get(model_name)
            plt.plot(fpr, tpr, label="{} ROC (area = {})".format(model_name, round(auc, 2)))

        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1-Specificity(False Positive Rate)')
        plt.ylabel('Sensitivity(True Positive Rate)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        plots_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Plots"))
        plt.savefig("{}/temp_{}.png".format(plots_path, image_index))
        pdf.image("{}/temp_{}.png".format(plots_path, image_index), w=175, h=105)
        image_index += 1
        pdf.ln(6)

    else:
        pdf.ln(2)
        plt.figure(figsize=(15, 10))
        sn.set(font_scale=2)
        
        for model_name in top_5_models:
            fpr = roc_fpr
            tpr = roc_tpr
            auc = roc_auc
            plt.plot(fpr, tpr, label="{} ROC (area = {})".format(model_name, round(auc, 2)))

        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1-Specificity(False Positive Rate)')
        plt.ylabel('Sensitivity(True Positive Rate)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        plots_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Plots"))
        plt.savefig("{}/temp_{}.png".format(plots_path, image_index))
        pdf.image("{}/temp_{}.png".format(plots_path, image_index), w=175, h=105)
        image_index += 1
        pdf.ln(6)

   
def model_selection(X, y, X_pred, donation_columns, cat_col, donor_df, no_donation_columns, skewed_target_value, skewed_target_value_similar, is_similar_file):

    # Handle Class imbalance using data augmentation techniques
    
    if ((skewed_target_value == True and is_similar_file == False) or (skewed_target_value_similar == True and is_similar_file == True)):
        over_sampling = RandomOverSampler(random_state=1, sampling_strategy=0.4)
        under_sampling = RandomUnderSampler(sampling_strategy=0.2, random_state=1)
        #define pipeline
        #steps = [('o', over_sampling), ('u', under_sampling)]
        steps = [('o', over_sampling)]
        pipeline = Pipeline(steps=steps)
        # Transform dataset
        X, y = pipeline.fit_resample(X, y)

    models = [{'label': 'LogisticRegression', 'model': LogisticRegression()},
                {'label': 'RidgeClassifier', 'model': RidgeClassifier()},  # No predict_proba
                {'label': 'MultinomialNB', 'model': MultinomialNB()},
                {'label': 'ComplementNB', 'model': ComplementNB(alpha=0.5, class_prior=None, fit_prior=True, norm=False)},
                {'label': 'BernoulliNB', 'model': BernoulliNB()},
                {'label': 'DecisionTreeClassifier', 'model': DecisionTreeClassifier()},
                {'label': 'SGDClassifier', 'model': SGDClassifier(loss='log')},
                {'label': 'PassiveAggressiveClassifier', 'model': PassiveAggressiveClassifier()},  # No predict_proba
                {'label': 'LinearSVC', 'model': LinearSVC()},  # No predict_proba
                {'label': 'RandomForestClassifier', 'model': RandomForestClassifier()}]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    model_f1_score = {}
    classification_full_pred = {}
    classification_full_pred_prob = {}
    feature_importance_dict = {}
    roc_fpr = {}
    roc_tpr = {}
    roc_auc = {}
    y_test_dict = {}
    y_pred_dict = {}
    
    for ind, m in enumerate(models):
        start_time = time.time()
        model = m['model']
        
        if m['label'] in ['PassiveAggressiveClassifier', 'LinearSVC', 'RidgeClassifier']:
            model = CalibratedClassifierCV(model)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        classification_full_pred[m['label']] = model.predict(X_pred)
        classification_full_pred_prob[m['label']] = model.predict_proba(X_pred)


        print("Classifier: {} and time(seconds): {}".format(m['label'], round(time.time() - start_time, 3)))
        print()

        model_f1_score[m['label']] = round(f1_score(y_test, y_pred, average='weighted'), 2)

        y_test_dict[m['label']] = y_test
        y_pred_dict[m['label']] = y_pred

        if m['label'] in ['DecisionTreeClassifier', 'RandomForestClassifier']:
            feature_value = model.feature_importances_[:-1]
        elif m['label'] in ['PassiveAggressiveClassifier', 'LinearSVC', 'RidgeClassifier']:
            model = m['model']
            model.fit(X_train, y_train)
            feature_value = model.coef_[0][:-1]
        elif m['label'] in ['GaussianNB']:
            continue
        else:
            feature_value = model.coef_[0][:-1]

        feature_importance_dict[m['label']] = feature_value

        fpr, tpr, auc = calculate_fpr_tpr(model, y_test, y_pred, X_test, None)
        roc_fpr[m['label']] = fpr
        roc_tpr[m['label']] = tpr
        roc_auc[m['label']] = auc

    #top_model = sorted(model_f1_score, key=model_f1_score.get,
    #reverse=True)[:1][0]
    top3_models = sorted(model_f1_score, key=model_f1_score.get, reverse=True)[:3]

    # ensemble the top 3 methods using the soft voting
    lst_models = list()
    for ind, m in  enumerate(models):

        if m['label'] in top3_models:

            if m['label'] in ['PassiveAggressiveClassifier', 'LinearSVC', 'RidgeClassifier']:
                lst_models.append((m['label'], CalibratedClassifierCV(m['model'])))
            else:
                lst_models.append((m['label'], m['model']))

    ensemble = VotingClassifier(estimators=lst_models, voting='soft')

    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    classification_full_pred['ensemble'] = ensemble.predict(X_pred)
    classification_full_pred_prob['ensemble'] = ensemble.predict_proba(X_pred)

    y_test_dict['ensemble'] = y_test
    y_pred_dict['ensemble'] = y_pred
    
    model_f1_score['ensemble'] = (round(f1_score(y_test, y_pred, average='weighted'), 2))

    fpr, tpr, auc = calculate_fpr_tpr(ensemble, y_test, y_pred, X_test, None)
    roc_fpr['ensemble'] = fpr
    roc_tpr['ensemble'] = tpr
    roc_auc['ensemble'] = auc

    feature_importance_dict['ensemble'] = np.mean(np.vstack((np.vstack((feature_importance_dict.get(lst_models[0][0]),  feature_importance_dict.get(lst_models[1][0]))),
                                                            feature_importance_dict.get(lst_models[2][0]))), axis=0)

    # PDF report section
    pdf.multi_cell(h=5.0, w=0, txt="     1. Model Name: Ensemble Method (Top 3 best fit classifiers)")
    pdf.ln(1)

    pdf.multi_cell(h=5.0, w=0, txt="              a. {}".format(top3_models[0]))
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="              b. {}".format(top3_models[1]))
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="              c. {}".format(top3_models[2]))
    pdf.ln(0.5)

    pdf.multi_cell(h=5.0, w=0, txt="     2. Total Data Sample: {}".format(convert_number_format(donor_df.shape[0])))

    if (not no_donation_columns) & (not is_similar_file) & (skewed_target_value == False):

        pdf.ln(1)
        pdf.multi_cell(h=5.0, w=0, txt="              a. 80% ({}) of data used for training the model".format(convert_number_format(X_train.shape[0])))
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="              b. 20% ({}) of data used for testing the model".format(convert_number_format(X_test.shape[0])))
        pdf.ln(0.5)
    if ((skewed_target_value == True and is_similar_file == False) or (skewed_target_value_similar == True and is_similar_file == True)):

        pdf.ln(1)
        pdf.multi_cell(h=5.0, w=0, txt="              a. 80% of data used for training the model")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="              b. 20% of data used for testing the model")
        pdf.ln(0.5)


    test_list = [chr(x) for x in range(ord('a'), ord('z') + 1)]
    if (no_donation_columns or is_similar_file == True):
        pdf.multi_cell(h=5.0, w=0, txt="     3. Donation Columns: The input data is missing donation information (no Target Value) OR only has one class (class")
                                      
        pdf.ln(0.25)
        pdf.multi_cell(h=5.0, w=0, txt="         donor or class non-donor 'data imbalance') to list donation column(s). In such cases, to execute the model")

        pdf.ln(0.25)
        pdf.multi_cell(h=5.0, w=0, txt="         successfully, data features from the input file are matched against data features on the similar donor files stored in")

        pdf.ln(0.25)
        pdf.multi_cell(h=5.0, w=0, txt="         the Predict Me's server. Donor file with the highest match rate is selected and processed using Natural Language")

        pdf.ln(0.25)
        pdf.multi_cell(h=5.0, w=0, txt="         Processing (NLP) method to maximize the model performance.")
                                   
        
        pdf.ln(0.5)
   
    else:
        pdf.multi_cell(h=5.0, w=0, txt="     3. Donation Columns:")
        pdf.ln(0.5)
        for i in range(len(donation_columns)):
            pdf.multi_cell(h=5.0, w=0, txt="              {}. {}".format(test_list[i], donation_columns[i]))
            pdf.ln(0.3)
        pdf.ln(0.5)

    if (len(cat_col) > len(test_list) and is_similar_file == False):
        cat_col = random.sample(cat_col, len(test_list))
    if (len(cat_col) != 0 and is_similar_file == False):
        pdf.multi_cell(h=5.0, w=0, txt="     4. Categorical Data Features:")
        pdf.ln(0.5)
        for i in range(len(cat_col)):
            pdf.multi_cell(h=5.0, w=0, txt="              {}. {}".format(test_list[i], cat_col[i]))
            pdf.ln(0.3)
    else:
        pdf.multi_cell(h=5.0, w=0, txt="     4. Categorical Data Features: No categorical data features processed from the input file due to missing ")
                                       

        pdf.ln(0.25)
        pdf.multi_cell(h=5.0, w=0, txt="          donation information OR only has one class to list categorical data features. In such cases, the donor file")

        pdf.ln(0.25)
        pdf.multi_cell(h=5.0, w=0, txt="          with the highest match rate is selected to calculate data features importance.")

        pdf.ln(0.5)
    
    
    pdf.set_font(font_style, 'BU', size=10)
    pdf.multi_cell(h=7.5, w=0, txt="B. Important Metrics Definition")
    pdf.set_font(font_style, size=10)
    pdf.ln(1)
    pdf.multi_cell(h=5.0, w=0, txt="     1. Text Data Conversion: Process of converting text data into a form that a model can understand.")
    pdf.ln(0.5)

    if(no_donation_columns == True or is_similar_file == True):

        pdf.multi_cell(h=5.0, w=0, txt="     2. Target Value: Total donation columns from the input file are calculated to assign Target Values. For example, in ")
                                            
        pdf.ln(0.25)
        pdf.multi_cell(h=5.0, w=0, txt="          each row, if 50% of the total donation columns have a donation amount >= 1, the model assigns that row as 1")
                                               
        pdf.ln(0.25)
        pdf.multi_cell(h=5.0, w=0, txt="          otherwise 0. Note: The input data is missing donation information OR only has one class. In such cases,")
                                              
        pdf.ln(0.25)
        pdf.multi_cell(h=5.0, w=0, txt="          the donor file with the highest match rate is selected to calculate Target Value.")

        pdf.ln(0.5)

    else:
        pdf.multi_cell(h=5.0, w=0, txt="     2. Target Value: Total donation columns from the input file are calculated to assign target values. For example, in ")
                                                
        pdf.ln(0.25)
        pdf.multi_cell(h=5.0, w=0, txt="         each row, if 50% of the total donation columns have a donation amount >=1, the model assigns that row as 1")
        pdf.ln(0.25)
        pdf.multi_cell(h=5.0, w=0, txt="         otherwise 0.")
        pdf.ln(0.5)

    pdf.multi_cell(h=5.0, w=0, txt="     3. Assign Target Value/Class (1 and 0): 1 = class donor or 1. 0 = class non-donor or 0.")
    pdf.ln(0.5)

    pdf.multi_cell(h=5.0, w=0, txt="     4. Data Imbalance: Skewness of the dataset.")
    pdf.ln(0.5)

    pdf.multi_cell(h=5.0, w=0, txt="     5. Training Set: Subset of data to train a model. Test set: Subset of data to test the trained model.")
    pdf.ln(0.5)

    pdf.multi_cell(h=5.0, w=0, txt="     6. Data Feature Importance: Process of selecting data features that contributes the most in prediction.")
    pdf.ln(0.5)

    pdf.multi_cell(h=5.0, w=0, txt="     7. Classifier/Model: Classifier or classification model is an algorithm that predicts classes.")
    pdf.ln(0.5)

    pdf.multi_cell(h=5.0, w=0, txt="     8. Ensemble Method: Technique of combining several models to generate one best model.")
    pdf.ln(0.5)

    pdf.multi_cell(h=5.0, w=0, txt="     9. Soft Voting: Process of generating best result by averaging all the predicted probabilities calculated by distinct ")
                                            
    pdf.ln(0.25)
    pdf.multi_cell(h=5.0, w=0, txt="         models.")
    pdf.ln(0.5)

    pdf.multi_cell(h=5.0, w=0, txt="     10. Performance Metrics: Metrics to explain the performance of a model.")
    pdf.ln(0.5)

    pdf.multi_cell(h=5.0, w=0, txt="     11. Precision: Fraction of correct predictions for a certain class. It refers to the percentage "
                                         "of results that are relevant.")
    pdf.ln(0.5)

    pdf.multi_cell(h=5.0, w=0, txt="     12. Recall: Fraction of correct predictions of all actual classes. It refers to the percentage of total relevant results")
                                   
    pdf.ln(0.25)
    pdf.multi_cell(h=5.0, w=0, txt="           correctly classified.")

    pdf.ln(0.5)

    pdf.multi_cell(h=5.0, w=0, txt="     13. F1- Score: Measure of a model's accuracy. A perfect model has an F1-score of 1.")
    pdf.ln(0.5)
    
    pdf.multi_cell(h=5.0, w=0, txt="     14. Confusion Matrix Plot: Visualized table describing the performance of a model. Each row in a confusion matrix")
    pdf.ln(0.25)
    pdf.multi_cell(h=5.0, w=0, txt="           represents an actual class, while each column represents a predicted class (or vice versa). Classification ")
    pdf.ln(0.25)
    pdf.multi_cell(h=5.0, w=0, txt="           report table is the performance metrics of a model.")
    pdf.ln(0.5)

    pdf.multi_cell(h=5.0, w=0, txt="     15. ROC Curve: Graph representing predicted performance of a model.")
    pdf.ln(0.5)
   
    pdf.multi_cell(h=5.0, w=0, txt="     16. Threshold: Cut-off value on a probability score to separate a donor from a non-donor. ")
    pdf.ln(0.5)

    pdf.multi_cell(h=5.0, w=0, txt="     17. Probability Score: Predicted probability (likelihood) score of an individual to donate.")
    pdf.ln(0.5)

    pdf.multi_cell(h=5.0, w=0, txt="     18. Predicted Classification (1 and 0): 1 indicates an individual likely to donate. 0 indicates "
                                   "an individual less likely to")
    pdf.ln(0.25)
    pdf.multi_cell(h=5.0, w=0, txt="           donate. They follow the threshold (cut-off) value logic.")
    pdf.ln(0.5)
    

    print_steps_taken(is_similar_file)

    plt.figure(figsize=(15, 10))

    
    return model_f1_score, classification_full_pred, classification_full_pred_prob, feature_importance_dict, roc_fpr, \
           roc_tpr, roc_auc, y_test_dict, y_pred_dict, top3_models

# Generate Prediction File with best classifier
def generate_prediction_file(df, model_f1_score, classification_full_pred, classification_full_pred_prob, y,
                             feature_importance_dict, roc_fpr, roc_tpr, roc_auc, y_test_dict, y_pred_dict,
                             feature_names, df_info, donation_columns_df, no_donations_columns, skewed_target_value, skewed_target_value_similar,
                             top3_models, is_similar_file):

    model_f1_score = {k: v for k, v in sorted(model_f1_score.items(), key=lambda item: item[1])}
    
    # Number of models we want in report, modify the count below
    # top_5_model = sorted(model_f1_score, key=model_f1_score.get,
    # reverse=True)[:1]

    # adding Assigned Target Value to the csv output when Similar file is not
    # used
    if(no_donations_columns == False or is_similar_file == False):
        df["Assigned Target Value"] = y


    top_5_model = ['ensemble'] # this method is the ensemble of top 3 models.  If only the top model is
                               # preferred, uncomment the top line for
                                                             # "top_5_model"

    pdf.set_font(font_style, 'BU', size=10)
    pdf.multi_cell(h=7.5, w=0, txt="D. Ensemble Method Output Metrics")
    pdf.set_font(font_style, size=10)
    pdf.ln(1)
    if(no_donations_columns == True or is_similar_file == True):
        pdf.multi_cell(h=5.0, w=0, align ='L', txt="Soft voting ensemble method used to combine the predictions of the top 3 best fit classifiers (models)."
                                                    " Following are F1-score, Threshold and Total Donors Predicted metrics.")
    else:
        pdf.multi_cell(h=5.0, w=0, align ='L', txt="Soft voting ensemble method used to combine the predictions of the top 3 best fit classifiers (models)."
                                                    " Following are the Assigned Target Value, F1-score, Threshold and Total Donors Predicted metrics.")

    for ind, m in enumerate(top_5_model):
        prediction = classification_full_pred.get(m)
        prob = classification_full_pred_prob.get(m)
        probability_column_name = 'Model Name: {}: Donor Probability Score'.format(m)
        prediction_column_name = 'Model Name: {}: Donor Predicted Classification (>= Threshold Value)'.format(m)
        df[probability_column_name] = [round(prob[x][1], 2) for x in range(len(prob))]
        # df['non_donor_prob_{}'.format(m)] = [round(prob[x][0], 2) for x in
        # range(len(prob))]
        if (no_donations_columns or is_similar_file == True):
            max_acc_threshold = [0.4]
        else:
            t_ = {}
            for t in [0.4, 0.45, 0.5, 0.55, 0.6]:   # [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
                df[prediction_column_name] = df[probability_column_name].apply(lambda x: 1 if x >= t else 0)
                t_[t] = round(f1_score(list(df[prediction_column_name]), y), 3)
            print(m)
            t_sorted = sorted(t_.items(), key=operator.itemgetter(1))
            print(t_sorted)
            max_acc_threshold = t_sorted[-1]

        print("Threshold used: {}".format(max_acc_threshold[0]))
        df[prediction_column_name] = df[probability_column_name].apply(lambda x: 1 if x >= max_acc_threshold[0] else 0)

        donor_count = df[df[prediction_column_name] == 1].shape[0]
        donor_per = round((donor_count / df.shape[0]) * 100, 2)
        #pdf.ln(2)
        
        pdf.set_font(font_style, size=10)
        pdf.ln(1)
        if(no_donations_columns == True or is_similar_file == True):
            pdf.multi_cell(h=5.0, w=0, txt="        a. F1-Score: {}".format(model_f1_score.get(m)))
            pdf.ln(0.75)
            pdf.multi_cell(h=5.0, w=0, txt="        b. Threshold: {}".format(max_acc_threshold[0]))
            pdf.ln(0.75)
            pdf.multi_cell(h=5.0, w=0, txt="        c. Donors Predicted: {}% ({} out of {})".format(donor_per, convert_number_format(donor_count), convert_number_format(df.shape[0])))

        else:
            pdf.multi_cell(h=5.0, w=0, txt="        a. Assigned Target Value (class donor): {}".format(convert_number_format(y.count(1))))
            pdf.ln(0.75)
            pdf.multi_cell(h=5.0, w=0, txt="        b. F1-Score: {}".format(model_f1_score.get(m)))
            pdf.ln(0.75)
            pdf.multi_cell(h=5.0, w=0, txt="        c. Threshold: {}".format(max_acc_threshold[0]))
            pdf.ln(0.75)
            pdf.multi_cell(h=5.0, w=0, txt="        d. Donors Predicted: {}% ({} out of {})".format(donor_per, convert_number_format(donor_count), convert_number_format(df.shape[0])))

        pdf.ln(3)
        print("F1-score: {}".format(model_f1_score.get(m)))
        print("Donors predicted: {}% ({} out of {})".format(donor_per, convert_number_format(donor_count), convert_number_format(df.shape[0])))

        # When the donation columns of the real input file exist (regardless of
        # similar file is used or not due to skewness), the evaluation metrics
        # should be used for the input file using the values in
        # df[prediction_column_name] & y
        if (no_donations_columns == False and len(y) != 0 and is_similar_file == False): 

            print_confusion_matrix_classification_report(y, df[prediction_column_name], no_donations_columns, skewed_target_value, is_similar_file)

            # calc roc_fpr, roc_tpr, roc_auc for the real input file
            fpr_inputfile, tpr_inputfile, auc_inputfile = calculate_fpr_tpr(None, y, df[prediction_column_name], None, df[probability_column_name])
            plot_roc_curve(fpr_inputfile, tpr_inputfile, auc_inputfile, top_5_model, no_donations_columns, skewed_target_value, is_similar_file)

        else: # donation columns does not exist
            print_confusion_matrix_classification_report(y_test_dict.get(m), y_pred_dict.get(m), no_donations_columns,
                                                     skewed_target_value, is_similar_file)
             
            plot_roc_curve(roc_fpr, roc_tpr, roc_auc, top_5_model, no_donations_columns, skewed_target_value, is_similar_file)

        calculate_feature_importance(df_info, feature_names, feature_importance_dict.get(m), no_donations_columns,
                                     skewed_target_value, skewed_target_value_similar, is_similar_file)

    # The following lines are commented since donation columns do not play any
    # role in prediction.  They just helped with determining target.

    #if donation_columns_df.shape[1] != 0:
    #    generate_correlation(donation_columns_df, no_donations_columns,
    #    skewed_target_value)

    return df, m


# Get tfidf featues for file found from DB (No donation columns present)
def get_tfidf_features(file_name):
    df = read_input_file(file_name)
    df = remove_rows_containg_all_null_values(df)
    df_info = remove_columns_unique_values(df, identify_info_columns(df, []))
    df_info = df_info.astype(str)
    df_info['comb_text'] = df_info.apply(lambda x: ' '.join(x), axis=1)
    processed_text = text_processing(list(df_info['comb_text']))
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_text)
    return vectorizer.get_feature_names()


# Find similar files in DB
def find_similar_files(input_file, no_donations_columns, skewed_target_value):
    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), input_file))
    directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "donation_amount_files"))
    input_features = get_tfidf_features(input_file)
    common_features = {}
    for file_name in glob.glob(directory_path + '/*.*'):
        if file_name != input_file:
            file_features = get_tfidf_features(file_name)
            print("Total features: {} for: {}".format(len(set(file_features)), file_name.split('/')[-1]))

            common_features_per = float(len(set(file_features) & set(input_features))) * 100 / len(set(input_features))
            common_features[file_name] = common_features_per 
            print("% of common features: {} for: {}".format(common_features_per, file_name))

    file_dict = {k: v for k, v in sorted(common_features.items(), key=lambda item: item[1])}
    x = sorted(file_dict, key=file_dict.get, reverse=True)

    # loop through list of similar files and measure the skewness for the top 5
    # files and select the file which is not skewed otherwise return null
    if len(x) == 0:
        return [], [], [], True, "", 
    elif len(x) < 5:
        threshold = len(x)
    else:
        threshold = 5

    file_index = []  # it holds the skewed similar file
    for i in range(threshold):
        df = read_input_file(x[i])
        df = remove_rows_containg_all_null_values(df)
        donation_columns = identify_years_columns(x[i])
        if (donation_columns != "[]"):
            donation_columns_df = process_donation_columns(df, donation_columns, no_donations_columns, skewed_target_value)
            no_donations_columns_new, skewed_target_value_new = check_skew_donation_cols(donation_columns_df, donation_columns)
            if skewed_target_value_new == False:
                print("The most similar file to the input file is {}: ".format(x[i]))
                return df, donation_columns, donation_columns_df, skewed_target_value_new, x[i].split('\\')[-1], common_features[x[i]]
            else: # similar file has donation columns but it is skewed
                file_index.append(i)
    
    if(len(file_index) != 0):

        df = read_input_file(x[file_index[0]])
        df = remove_rows_containg_all_null_values(df)
        donation_columns = identify_years_columns(x[file_index[0]])
        donation_columns_df = process_donation_columns(df, donation_columns, no_donations_columns, skewed_target_value)
        if (donation_columns != "[]"):
            print("The most similar file to the input file is {}: ".x[file_index[0]])
            return df, donation_columns, donation_columns_df, True, x[file_index[0]], common_features[x[file_index[0]]]

    return [], [], [], True, None, None


# Transform text values to tf-idf features
def transform_features(vectorizer, df_info):
    df_info = df_info.astype(str)
    df_info['comb_text'] = df_info.apply(lambda x: ' '.join(x), axis=1)
    processed_text = text_processing(list(df_info['comb_text']))
    X = vectorizer.transform(processed_text)
    tfidf_matrix = X.todense()
    return tfidf_matrix


# Print steps taken to run classifier in PDF
def print_steps_taken(is_similar_file):
    pdf.ln(1)
    pdf.set_font(font_style, 'BU', size=10)
    pdf.multi_cell(h=7.5, w=0, txt="C. Steps on Building and Executing Predictive Model")
    pdf.set_font(font_style, size=10)
    pdf.ln(1)
    pdf.multi_cell(h=5.0, w=0, txt="     1. Read the input data file.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     2. Data cleanse. Impute missing values and remove null rows and columns.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     3. Convert text data to numbers.")
    pdf.ln(0.5)

    if(is_similar_file == True):
        pdf.multi_cell(h=5.0, w=0, txt="     4. Assign Target Value. Target values are the predicted variable. Since the uploaded donor file is missing donation")
                                                

        pdf.ln(0.25)
        pdf.multi_cell(h=5.0, w=0, txt="         information (amount), target values cannot be calculated.")
        pdf.ln(0.5)

    else:
        pdf.multi_cell(h=5.0, w=0, txt="     4. Assign Target Value. Target values are the predicted variable.")
        pdf.ln(0.5)

    if ( (skewed_target_value == True and  is_similar_file==False) or (skewed_target_value_similar == True and  is_similar_file==True)):

        pdf.multi_cell(h=5.0, w=0, txt="     5. Evaluate data imbalance.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="     6. Over and Under Sampling techniques used.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="     7. Split data for training and testing.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="     8. Calculate Data Feature Importance.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="     9. Evaluate performance metrics of 10 classifiers (models) and select top 3 best fit classifiers.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="     10. Combine top 3 best fit classifiers predictions using a soft voting ensemble method.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="     11. Create Confusion Matrix, Classification Report and Receiver Operating Characteristic (ROC) Curve.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="     12. Identify the threshold and predict.")
        pdf.ln(0.5)
    

    else:

        pdf.multi_cell(h=5.0, w=0, txt="     5. Evaluate data imbalance.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="     6. Split data for training and testing.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="     7. Calculate Data Feature Importance.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="     8. Evaluate performance metrics of 10 classifiers (models) and select top 3 best fit classifiers.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="     9. Combine top 3 best fit classifiers predictions using a soft voting ensemble method.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="     10. Create Confusion Matrix, Classification Report and Receiver Operating Characteristic (ROC) Curve.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="     11. Identify the threshold and predict.")
        pdf.ln(0.5)
    
 
    if(is_similar_file == True):
        pdf.multi_cell(h=5.0, w=0, txt="     12. Generate model summary report (PDF) and CSV file with Donor Probability Score and Donor Predicted ")

        pdf.ln(0.25)
        pdf.multi_cell(h=5.0, w=0, txt="           Classification columns appended back to the processed file.")
    else:
        pdf.multi_cell(h=5.0, w=0, txt="     13. Generate model summary report (PDF) and CSV file with the Assigned Target Value, Donor Probability Score ")

        pdf.ln(0.25)
        pdf.multi_cell(h=5.0, w=0, txt="           and Donor Predicted Classification columns appended back to the processed file.")

    
    pdf.ln(3)


# Delete old plots from directory
def delete_old_plots():
    plots_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Plots"))
    files = glob.glob('{}/*.png'.format(plots_path))
    for f in files:
        os.remove(f) 

if __name__ == "__main__":
    
    start_time = time.time()
    pdf.ln(2)
    today = date.today()
    today_date = today.strftime("%B %d, %Y")
    pdf.set_font(font_style, 'B', size=10)
    pdf.multi_cell(h=5.0, w=0, txt="Predictive Modeling Results", align="C")
    pdf.ln(0.5)
    pdf.set_font(font_style, 'B', size=8)
    pdf.multi_cell(h=5.0, w=0, txt="Report Date: {}".format(today_date), align="C")
    pdf.ln(3)
    delete_old_plots()

    file_path = sys.argv[1]
    donation_columns = ast.literal_eval(sys.argv[2])

    donor_df = read_input_file(file_path)

    pdf.set_font(font_style, 'BU', size=10)
    pdf.multi_cell(h=7.5, w=0, txt="A. Data Input Summary")
    pdf.set_font(font_style, size=10)
    pdf.ln(1)
    donor_df = remove_rows_containg_all_null_values(donor_df)
    no_donations_columns = False
    skewed_target_value = False
    is_similar_file = False
    skewed_target_value_similar = False
    similar_filename = None
    common_features = None
    y = []  # list of target values

    # check if donation_columns exist
    if(len(donation_columns) != 0):
        donation_columns_df = process_donation_columns(donor_df, donation_columns, no_donations_columns,  skewed_target_value)
        # check for skewness
        no_donations_columns , skewed_target_value = check_skew_donation_cols(donation_columns_df, donation_columns)

        # when there is only one class in the file, "no_donations_columns"
        # becomes true
        if(no_donations_columns == True):
            df, donation_columns_similar, donation_columns_df_similar, skewed_target_value_similar, similar_filename, common_features = find_similar_files(file_path, no_donations_columns , skewed_target_value)
            if (len(df) == 0):
                raise ValueError("The similar file is not found, and the donation columns of input file does not exist. "
                             "Please provide donation columns on your donor file.")

            else: # if similar file found
                is_similar_file = True
                donation_columns_df = []
        else:
            skewed_target_value_similar = False
            df = donor_df

    # donation columns of input file do not exist.  So, find the similar file
    # Note: When donation columns of input file do not exist, skewness cannot
    # be measured.
    else: 
        no_donations_columns = True
        df, donation_columns_similar, donation_columns_df_similar, skewed_target_value_similar, similar_filename, common_features = find_similar_files(file_path, no_donations_columns , skewed_target_value)
        
        # if the similar file Not found
        if (len(df) == 0):
            raise ValueError("The similar file is not found, and the donation columns of input file does not exist.  "
                             "Please provide donation columns on your donor file.")

        else: # if similar file found
            is_similar_file = True
            donation_columns_df = []
           
#********************************************************************************************
    
    if (is_similar_file):
        info_columns = identify_info_columns(df, donation_columns_similar)
        y_similar = list(donation_columns_df_similar['target'])
        if no_donations_columns == False:
            y = list(donation_columns_df['target'])
    else:
        info_columns = identify_info_columns(df, donation_columns)
        y = list(donation_columns_df['target'])
        y_similar = y

    df_info = remove_columns_unique_values(df, info_columns)
    if len(info_columns) < 3:
        raise ValueError("In order for the model to run, please supply a minimum of three text columns on your donor "
                         "file.")
    
    processed_text, tfidf_matrix, feature_names, df_info, vectorizer = feature_extraction(df_info)
        
    if (is_similar_file and no_donations_columns == False): # Similar file is used (donation columns exist)
        input_file_text_cols = identify_info_columns(donor_df, donation_columns)
        donor_df_text = donor_df[input_file_text_cols]
        X_pred = transform_features(vectorizer, donor_df_text)
        X_train = tfidf_matrix

    elif(is_similar_file and no_donations_columns): # Simialr file is used but donation columns do not exist
        X_pred = transform_features(vectorizer, donor_df)
        X_train = tfidf_matrix

    else:
        X_pred = tfidf_matrix
        X_train = tfidf_matrix
    
    if(is_similar_file == True and similar_filename != None and common_features != None):
        print("The most similar file to the input file is: {} \n"
              "% of common features in original input file and similar file is: {:.2f} \n"
              "Categorical columns of the similar file used for prediction models are: {}".format(similar_filename, common_features,
                                                                                            ", ".join(df_info.columns[0:-1])))

    model_f1_score, classification_full_pred, classification_full_pred_prob, feature_importance_dict, roc_fpr, \
    roc_tpr, roc_auc, y_test_dict, y_pred_dict, top3_models = model_selection(X_train, y_similar, X_pred, donation_columns, df_info.columns[0:-1], donor_df,
                                                                 no_donations_columns, skewed_target_value, skewed_target_value_similar, is_similar_file)

    df_final, best_model = generate_prediction_file(donor_df, model_f1_score, classification_full_pred,
                                        classification_full_pred_prob, y, feature_importance_dict, roc_fpr, roc_tpr,
                                        roc_auc, y_test_dict, y_pred_dict, feature_names, df_info, donation_columns_df,
                                        no_donations_columns, skewed_target_value, skewed_target_value_similar, top3_models, is_similar_file)

    prediction_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "prediction"))
    pdf_report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "PdfReport"))
    file_path = file_path.split("/")[-1]
    pdf.output("{}/{}_{}_{}_report.pdf".format(pdf_report_path, file_path.split(".")[0], best_model, today_date))
    df_final.to_csv("{}/{}_{}_{}_prediction.csv".format(prediction_path, file_path.split(".")[0], best_model, today_date), index=None)

    print("total run time is {}".format(round(time.time() - start_time, 3)))
