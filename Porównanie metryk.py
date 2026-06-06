import pandas as pd
m_c=pd.read_csv('metryki_c_test.csv', sep=';')
m_c_clin=pd.read_csv('metryki_c_clin_test.csv', sep=';')
m_rf=pd.read_csv('metryki_rf_test.csv', sep=';')
m_rf_clin=pd.read_csv('metryki_rf_clin_test.csv', sep=';')
m_xgb=pd.read_csv('metryki_xgb_test.csv', sep=';')
m_xgb_clin=pd.read_csv('metryki_xgb_clin_test.csv', sep=';')
m_l=pd.read_csv('metryki_los.csv', sep=';')

metryki=[m_c, m_c_clin, m_rf, m_rf_clin, m_xgb, m_xgb_clin, m_l]
metryki[0]
Dokładność=[]
f1score=[]
Precyzja_przeżywalność=[]
for i in range(len(metryki)):
    d=metryki[i].iloc[3,3]
    f=metryki[i].iloc[5,3]
    pp=metryki[i].iloc[2,1]
    Dokładność.append(d)
    f1score.append(f)
    Precyzja_przeżywalność.append(pp)

modele=['CatBoost','CatBoost kliniczny', 'Random Forest', 'Random Forest kliniczny', 'XG Boost', 'XG Boost kliniczny', 'Losowy']

Porównanie_metryk=pd.DataFrame({
    'Model':modele,
    'Dokladnosc': Dokładność,
    'F1-score':f1score,
    'Precyzja przezywalnosci': Precyzja_przeżywalność
})

# Porównanie_metryk.to_csv('Porównanie_metryk.csv', sep=';')

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('AGG')

plt.figure(figsize=(10,4))
sns.heatmap(Porównanie_metryk.set_index('Model'), annot=True, fmt='.2f', cmap='Blues')
plt.title('Porownanie metryk')
plt.tight_layout()
plt.savefig('Porównanie_metryk.png')
plt.close()


from adjustText import adjust_text

plt.figure(figsize=(12, 8))

sns.scatterplot(
    data=Porównanie_metryk,
    x='Dokladnosc',
    y='F1-score',
    hue='Model',
    s=10,
    alpha=1
)
plt.tight_layout()
plt.savefig('Metryki.png')
plt.close()