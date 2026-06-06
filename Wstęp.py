from random import Random

import pandas
import  pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import kagglehub

# Download latest version



pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)
dane=pd.read_excel("Zeszyt1.xlsx")
dane.dtypes
dane[dane.select_dtypes(include="string").columns]=dane[dane.select_dtypes(include="string").columns].astype("category")
dane.replace(("NA","N/A","NAN"), value=pd.NA)
dane["tumor_stage"]=dane["tumor_stage"].astype("category")
dane["chemotherapy"]=dane["chemotherapy"].astype("category")
dane["cohort"]=dane["cohort"].astype("category")
dane["neoplasm_histologic_grade"]=dane["neoplasm_histologic_grade"].astype("category")
dane["hormone_therapy"]=dane["hormone_therapy"].astype("category")
dane["overall_survival"]=dane["overall_survival"].astype("category")
dane["radio_therapy"]=dane["radio_therapy"].astype("category")
dane["death_from_cancer"]=dane["death_from_cancer"].astype("category")
dane['nottingham_prognostic_index']=dane['nottingham_prognostic_index'].astype('category')
dane['age_group'] = pd.cut(
    dane['age_at_diagnosis'],
    bins=[20, 40, 60, 100],
    labels=['20-40', '40-60', '60+']
)


zmienne_kliniczne=list(dane.columns[:31]) +['age_group']
zmienne_genowe=dane.columns.difference(zmienne_kliniczne)

