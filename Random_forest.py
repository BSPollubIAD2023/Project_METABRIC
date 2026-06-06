import  pandas as pd
from IPython.core.pylabtools import figsize
from PIL.ImageChops import difference
from sklearn.compose import ColumnTransformer

dane3=pd.read_csv('dane3.csv', sep=';')

import pickle
with open("eda_variables.pkl", "rb") as f:
    eda_vars = pickle.load(f)

zmienne_do_boxow = eda_vars["zmienne_do_boxow"]
zmienne_genowe = eda_vars["zmienne_genowe"]
zmienne_kliniczne = eda_vars["zmienne_kliniczne"]
kategoryczne_zmienne = eda_vars["kategoryczne_zmienne"]
numeryczne_zmienne = eda_vars["numeryczne_zmienne"]
kolumny_mar = eda_vars["kolumny_mar"]
kolumny_mcar = eda_vars["kolumny_mcar"]
zmienne_do_pca = eda_vars["zmienne_do_pca"]


#split
#↓
#imputacja na train
#↓
#ICA na train
#↓
#transformacja test
#↓
#model

target='death_from_cancer'

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

encoder=OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1
)
wynik_ist=pd.read_csv('wynik_po_korekcie.csv', sep=';')


mar=wynik_ist['missing in:'].unique()
from sklearn.preprocessing import OneHotEncoder
mar_num = [
    col for col in mar
    if not col.endswith('_mut')
    and col in numeryczne_zmienne
]
mar_num=[
    col for col in mar_num
    if col!='Unnamed: 0'
]

mar_cat=[
    col for col in mar
    if col in kategoryczne_zmienne
    or col.endswith('_mut')
]

mcar_num = [
    col for col in dane3.columns
    if not col.endswith('_mut')
    and col not in kategoryczne_zmienne
    and col not in mar
]
mcar_num = [
    col for col in mcar_num
    if col not in [
        'patient_id',
        'mahalanobis',
        'therapy_combo',
        'outlier',
        'age_group',
        'death_from_cancer',
        'Unnamed: 0'

    ]
]

mcar_num=mcar_num +['nottingham_prognostic_index']
mcar_cat=[
    col for col in dane3.columns
    if col not in mar
    and col in kategoryczne_zmienne
]
mcar_cat = [
    col for col in mcar_cat
    if col not in [
        'patient_id',
        'mahalanobis',
        'chemotherapy',
        'radio_therapy',
        'hormone_therapy',
        'age_group',
        'nottingham_prognostic_index',
        'death_from_cancer',
        'overall_survival'
    ]
]




mcar_num = [
    col for col in mcar_num
    if col not in zmienne_do_pca
]
mcar_cat = [
    col for col in mcar_cat
    if col not in zmienne_do_pca
]
mar_num = [
    col for col in mar_num
    if col not in zmienne_do_pca
]
mar_cat = [
    col for col in mar_cat
    if col not in zmienne_do_pca
]
mar_cat=[
    col for col in mar_cat
    if col !=target
]





X=dane3.copy()
X[mcar_cat]=X[mcar_cat].astype('category')
X[mar_cat]=X[mar_cat].astype('category')
X=X.drop(columns=target)
X=X.drop(columns='patient_id')
X=X.drop(columns='overall_survival')
X=X.drop(columns='age_group')
X=X.drop(columns='mahalanobis')
X=X.drop(columns=['chemotherapy',
        'radio_therapy',
        'hormone_therapy'])
X=X.drop(columns='Unnamed: 0')
y=dane3[target]
maska=y.notna()
y=y[maska]
X=X[maska]


X_train_val, X_test, y_train_val, y_test=train_test_split(X,y,test_size=0.2,random_state=2026, stratify=y )
zmienne_do_pca=list(zmienne_do_pca)
X_train, X_val, y_train, y_val=train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=2026, stratify=y_train_val)
X_train_geny = X_train[zmienne_do_pca]
X_val_geny=X_val[zmienne_do_pca]
X_test_geny = X_test[zmienne_do_pca]

X_train_reszta = X_train.drop(columns=zmienne_do_pca)
X_val_reszta=X_val.drop(columns=zmienne_do_pca)
X_test_reszta = X_test.drop(columns=zmienne_do_pca)

from sklearn.pipeline import Pipeline

gen_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_train_geny = gen_pipe.fit_transform(X_train_geny)
X_val_geny=gen_pipe.transform(X_val_geny)
X_test_geny = gen_pipe.transform(X_test_geny)
from sklearn.decomposition import FastICA
ica = FastICA(
    n_components=60,
    random_state=2026
)

X_train_ica = ica.fit_transform(X_train_geny)
X_val_ica=ica.transform(X_val_geny)
X_test_ica = ica.transform(X_test_geny)



mcar_num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

mar_num_pipe = Pipeline([
    ("imputer", IterativeImputer(
        random_state=2026,
        max_iter=20
    ))
])

mcar_cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ('encoder', encoder)
])

mar_cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ('encoder', encoder)
])


preprocessor = ColumnTransformer(
    transformers=[
        ("mcar_num", mcar_num_pipe, mcar_num),
        ("mar_num", mar_num_pipe, mar_num),
        ("mcar_cat", mcar_cat_pipe, mcar_cat),
        ("mar_cat", mar_cat_pipe, mar_cat)
    ],
    remainder="drop"
)
X_train_reszta=preprocessor.fit_transform(X_train_reszta)
X_val_reszta=preprocessor.transform(X_val_reszta)
X_test_reszta=preprocessor.transform(X_test_reszta)


import numpy as np
X_train_final = np.hstack([
    X_train_reszta,
    X_train_ica
])

X_val_final=np.hstack([
    X_val_reszta,
    X_val_ica
])

X_test_final = np.hstack([
    X_test_reszta,
    X_test_ica
])

ica_cols = [f"IC{i+1}" for i in range(60)]
num=mcar_num+mar_num + ica_cols
cat=mcar_cat+mar_cat


cols = (
    mcar_num +
    mar_num +
    mcar_cat +
    mar_cat+
    ica_cols
)


#
# from sklearn.ensemble import RandomForestClassifier
#
# rf = RandomForestClassifier(
#     n_estimators=1000,
#     max_depth=8,
#     min_samples_split=5,
#     min_samples_leaf=3,
#     random_state=2026,
#     class_weight='balanced',
#     n_jobs=-1,
#     verbose=1
# )
# rf.fit(X_train_final, y_train)
# y_pred_val=rf.predict(X_val_final)
#
# from sklearn.metrics import classification_report
#
# classification_report_rf=classification_report(
#     y_val,
#     y_pred_val,
#     output_dict=True
# )
# classification_report_rf = pd.DataFrame(
#     classification_report_rf
# ).T
# classification_report_rf.to_csv('metryki_rf.csv', sep=';')
#
# parametry_rf=rf.get_params()
#
# parametry_rf=pd.DataFrame([parametry_rf]).T
# parametry_rf.to_csv('parametry_rf.csv', sep=';')
#
#
#
#
# wagi_cech_rf=pd.DataFrame({
#     'cecha': cols,
#     'wagi':rf.feature_importances_
# }).sort_values('wagi', ascending=False)
#
# wagi_cech_rf.to_csv('wagi_cech_rf.csv', sep=';')
#
# import pickle
#
# with open("random_forest.pkl", "wb") as f:
#     pickle.dump(rf, f)

with open('random_forest.pkl', 'rb') as f:
    model=pickle.load(f)

from sklearn.metrics import classification_report

y_pred_test=model.predict(X_test_final)

classification_report_rf_test=classification_report(y_test, y_pred_test, output_dict=True)

classification_report_rf_test=pd.DataFrame(classification_report_rf_test).T
classification_report_rf_test.to_csv('metryki_rf_test.csv', sep=';')