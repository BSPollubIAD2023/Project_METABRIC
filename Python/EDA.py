#braki
from statistics import covariance
from xml.etree.ElementInclude import include

import matplotlib
import pandas as pd
import matplotlib
matplotlib.use('AGG')
from Wstęp import zmienne_kliniczne
from Wstęp import dane
from Wstęp import zmienne_genowe
pd.set_option('display.float_format', '{:.2f}'.format)
braki_counts=dane.isna().sum().sort_values(ascending=False)
braki_counts=braki_counts[braki_counts>0]
braki_percent=(dane.isna().mean().sort_values(ascending=False))*100
braki_percent=braki_percent[braki_percent>0]
braki_danych=pd.DataFrame(
    {
        "liczba_brakow":braki_counts,
        "procent_brakow":braki_percent
    }
)
braki_danych.to_csv("braki_danych.csv", sep=";")
dane.isna().any(axis=0).mean() #1% niepełne kolumny
dane.isna().any(axis=1).mean() #60% niepełnych wierszy

kategoryczne_zmienne=dane.select_dtypes(include="category").columns
numeryczne_zmienne=dane.columns.difference(kategoryczne_zmienne)

#charakter braków
from scipy.stats import chi2_contingency
kl_z_brakami=braki_counts[braki_counts>braki_counts.quantile(0.25)]
wynik=[]
for target in dane[kl_z_brakami.index]:
    target_na=dane[target].isna()
    for k in kategoryczne_zmienne:
        if k==target:
            continue
        crstab=pd.crosstab(dane[k], target_na)
        if (crstab.shape[0] > 1) & (crstab.shape[1] > 1):
            chi2_stat, p, dof, expected = chi2_contingency(crstab)
            wynik.append({
                "missing in:": target,
                "chi2:": chi2_stat,
                "p_value": p,
                "category": k

            })
wynik=pd.DataFrame(wynik).sort_values("p_value")
wynik.to_csv("wynik.csv", sep=";")


# patient_id – ID pacjenta

# age_at_diagnosis – wiek pacjenta w momencie diagnozy

# type_of_breast_surgery – typ operacji piersi:
# 1. MASTECTOMY – usunięcie całej piersi
# 2. BREAST CONSERVING – operacja oszczędzająca pierś (usunięcie tylko zmienionej części)

# cancer_type – typ nowotworu piersi:
# 1. Breast Cancer – rak piersi
# 2. Breast Sarcoma – mięsak piersi

# cancer_type_detailed – szczegółowy typ raka piersi:
# 1. Inwazyjny rak przewodowy
# 2. Mieszany rak przewodowy i zrazikowy
# 3. Inwazyjny rak zrazikowy
# 4. Inwazyjny mieszany rak śluzowy
# 5. Metaplastyczny rak piersi

# cellularity – komórkowość guza po chemioterapii
# (ilość komórek nowotworowych i ich rozmieszczenie)

# chemotherapy – czy pacjent otrzymał chemioterapię (tak/nie)

# pam50_+_claudin-low_subtype – podtyp molekularny PAM50 / claudin-low
# określający profil ekspresji genów nowotworu

# cohort – grupa badawcza pacjenta (wartości od 1 do 5)

# er_status_measured_by_ihc – status receptorów estrogenowych
# mierzony metodą immunohistochemii (pozytywny/negatywny)

# er_status – status receptorów estrogenowych komórek nowotworowych
# (pozytywny/negatywny)

# neoplasm_histologic_grade – stopień histologiczny nowotworu
# oceniany przez patologa (1–3)

# her2_status_measured_by_snp6 – status HER2 mierzony metodą SNP6
# (zaawansowane techniki molekularne)

# her2_status – status HER2 (pozytywny/negatywny)

# tumor_other_histologic_subtype – histologiczny podtyp nowotworu:
# 'Ductal/NST', 'Mixed', 'Lobular', 'Tubular/cribriform',
# 'Mucinous', 'Medullary', 'Other', 'Metaplastic'

# hormone_therapy – czy pacjent otrzymał terapię hormonalną (tak/nie)

# inferred_menopausal_state – status menopauzalny pacjenta
# (pre/post menopausal)

# integrative_cluster – molekularny klaster nowotworu
# oparty na ekspresji genów

# primary_tumor_laterality – strona występowania nowotworu
# (lewa/prawa pierś)

# lymph_nodes_examined_positive – liczba zajętych węzłów chłonnych

# mutation_count – liczba istotnych mutacji genetycznych

# nottingham_prognostic_index – indeks prognostyczny Nottingham
# określający rokowanie po operacji raka piersi

# oncotree_code – kod klasyfikacji nowotworu OncoTree

# overall_survival_months – czas przeżycia całkowitego w miesiącach

# overall_survival – zmienna docelowa:
# czy pacjent żyje czy zmarł

# pr_status – status receptorów progesteronowych
# (pozytywny/negatywny)

# radio_therapy – czy pacjent otrzymał radioterapię (tak/nie)

# 3-gene_classifier_subtype – podtyp klasyfikatora 3-genowego:
# 'ER-/HER2-', 'ER+/HER2- High Prolif',
# 'ER+/HER2- Low Prolif', 'HER2+'

# tumor_size – rozmiar guza

# tumor_stage – stadium nowotworu

# death_from_cancer – czy śmierć pacjenta była spowodowana nowotworem (tak/nie)
from statsmodels.stats import multitest
wynik['p_adj']=multitest.multipletests(
    wynik['p_value'],
    method="fdr_bh"
)[1]
pd.set_option("display.float_format", lambda x: f"{x:.3e}")
wynik_ist=wynik[wynik['p_adj']<=0.05]
wynik_ist.to_csv("wynik_po_korekcie.csv", sep=";")
#najwięcej braków danych w stadium rozwoju guza
kolumny_mar=wynik_ist['missing in:'].unique()
kolumny_mcar=braki_counts.index.unique()
# trzeba zrobić typ ~~~ braków
typy=[]
for m,c in zip(wynik_ist['missing in:'], wynik_ist['category']):
    if m in zmienne_kliniczne and c in zmienne_kliniczne:
        typy.append('clinic-clinic')
    elif m in zmienne_kliniczne and c in zmienne_genowe:
        typy.append('clinic-gene')
    elif m in zmienne_genowe and c in zmienne_genowe:
        typy.append('gene-gene')
    elif m in zmienne_genowe and c in zmienne_kliniczne:
        typy.append('gene-clinic')
wynik_ist['type']=typy
wynik_ist.to_csv("wynik_po_korekcie.csv", sep=";")

import seaborn as sns
import matplotlib.pyplot as plt

cormat=dane[zmienne_kliniczne].select_dtypes(exclude='category').drop(columns='patient_id').corr()

plt.figure(figsize=(21,18))
sns.heatmap(cormat, annot=True, fmt='.2f')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.savefig("korelacja_kli.png")
plt.close()
zmienne_kliniczne
#target w miarę zbalansowany

#chemioterapia, hormonalna terapia, typ operacji, typ raka szczegolowy, wiek, ilosc komorek nowotworowych, radioterapia, survival months, death from cancer
kombinacje=dane[
    ['chemotherapy',
    'hormone_therapy',
    'radio_therapy',
    'type_of_breast_surgery',
     'cancer_type_detailed',
     'age_group']
].value_counts()
kombinacje.to_csv("kombinacje.csv", sep=';')
dane['cancer_type_detailed'].value_counts() #jeden typ nowotworu domuinuje nad innymi jeden przypadek o liczności 1
dane['age_group'].value_counts() #głownie ludzie 60+ trochę mniej 40-60 mało 20-40


dane2=dane.copy()

dane2['therapy_combo']=(
    'C' + dane2['chemotherapy'].astype("str")+
    '_H' +dane2['hormone_therapy'].astype("str")+
    '_R' +dane2['radio_therapy'].astype('str')
)

plt.figure(figsize=(20,20))
g = sns.catplot(
    data=dane2,
    x='therapy_combo',
    y='overall_survival_months',
    hue='death_from_cancer',
    col='type_of_breast_surgery',
    row='age_group',
    kind='box',
    height=5,
    aspect=1.4
)
g._legend.set_bbox_to_anchor((1.01, 0.965))
for ax in g.axes.flat:
    ax.tick_params(axis='x', rotation=90)

plt.tight_layout()

plt.savefig("fig.png", dpi=300)

numeryczne_zmienne2=set(numeryczne_zmienne)
zmienne_kliniczne2=set(zmienne_kliniczne)
zmienne_do_boxow=numeryczne_zmienne2.intersection(zmienne_kliniczne2)
zmienne_do_boxow=list(zmienne_do_boxow)

for i, c in enumerate(zmienne_do_boxow, start=1):

    plt.figure(figsize=(8,4))

    sns.boxplot(y=dane2[c])
    plt.title(f'rozkład {c}')
    plt.tight_layout()
    plt.savefig(f'rozkład {c}.png')

    plt.close()
pd.set_option('display.float_format', '{:.0f}'.format)
dane[zmienne_do_boxow].describe()



opis = dane[zmienne_do_boxow] \
    .drop(columns='patient_id') \
    .describe() \
    .round(2)


plt.figure(figsize=(10,6))

sns.heatmap(
    opis,
    annot=True,
    fmt='.2f',
    cmap='Blues'
)
plt.xticks(rotation=60)
plt.title('Statystyki opisowe zmiennych numerycznych')

plt.tight_layout()

plt.savefig('opisowe_heatmap.png')

plt.close()

import  numpy as np
from scipy.spatial.distance import  mahalanobis
from sklearn.preprocessing import StandardScaler


X = dane2[numeryczne_zmienne].copy()

# zachowanie indeksów complete cases
idx = X.dropna().index

# complete cases
X_complete = X.loc[idx]

# scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_complete)

# macierz kowariancji
cov = np.cov(X_scaled, rowvar=False)

# pseudoodwrotność
cov_inv = np.linalg.pinv(cov)

# wektor średnich
mean = X_scaled.mean(axis=0)

# odległości mahalanobisa
distances = []

for row in X_scaled:

    d = mahalanobis(row, mean, cov_inv)

    distances.append(d)

# dataframe z dystansami
mahal_df = pd.DataFrame({
    'mahalanobis': distances
}, index=idx)

# dodanie do oryginalnych danych
dane2 = dane2.join(mahal_df)
from scipy.stats import chi2
# threshold chi-square
threshold = chi2.ppf(
    0.99,
    df=len(numeryczne_zmienne)
)

# outliery
dane2['outlier'] = dane2['mahalanobis']**2 > threshold
dane3=dane2.copy()
dane3.to_csv('dane3.csv', sep=";")

# liczba outlierów
print(dane2['outlier'].value_counts())


#1. Braki
#2. Mechanizm braków
#3. Balans zmiennej wynikowej
#4. Korelacje klinicznych
#5. Kombo terapii ze względu n a typ operacji, raka, grupy wiekowej, prognozowanych miesiecy przeżycia i samej zmiennej wynikowej
#6. Rozkłady zmiennych heatmapa
#7. Wielowymiarowe outliery

#Pca
#skalowanie i imputacja zmiennych genowych
kategoryczne_zmienne3=set(kategoryczne_zmienne)
zmienne_genowe3=set(zmienne_genowe)
zmienne_numeryczne3=set(numeryczne_zmienne)
zmienne_do_pca=zmienne_genowe3.intersection(zmienne_numeryczne3)
zmienne_do_pca=list(zmienne_do_pca)
zmienne_do_pca=zmienne_do_pca = [
    c for c in zmienne_do_pca
    if dane2[c].nunique(dropna=True) >= 3
]
dane_pca=dane2.copy()
dane_pca=dane_pca[zmienne_do_pca]
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='median')
dane_pca=imputer.fit_transform(dane_pca)
dane_pca_scaled=scaler.fit_transform(dane_pca)

from sklearn.decomposition import PCA
pca=PCA()
pca.fit(dane_pca_scaled)
plt.figure(figsize=(10,4))
cum_var=np.cumsum(pca.explained_variance_ratio_)
plt.plot(
    np.cumsum(pca.explained_variance_ratio_)
)
plt.title('Zsumowana wariancja wyjaśniona')
plt.xticks(range(0, len(cum_var)+1, 20))
plt.savefig('PCA_wariancja_wyj.png')
plt.close()

#nieskuteczne

from sklearn.decomposition import FastICA
ICA=FastICA()
from scipy.stats import skew, kurtosis
import pingouin as pg
pg.multivariate_normality(
    dane_pca_scaled,
    alpha=0.05
) #wielowymiarowy rozkład normalny ??

sk = skew(dane_pca_scaled, axis=0)
ku = kurtosis(dane_pca_scaled, axis=0)
import statistics
statistics.mode(ku)
pd.Series(sk).describe() #lekko prawoskośny
pd.Series(ku).describe() #silnie prawoskośny
np.sum(np.var(dane_pca_scaled, axis=0) == 0) #brak stałych kolumn

n_comp = np.argmax(cum_var >= 0.8)+1


ica = FastICA(
    n_components=n_comp,
    random_state=2026,
    max_iter=5000
)
X_ica=ica.fit_transform(dane_pca_scaled)
X_ica

ica.n_iter_
loadings = pd.DataFrame(
    ica.components_.T,
    index=zmienne_do_pca
)
ind_comp={}
pd.set_option('display.float_format', '{:.6f}'.format)
for i in range(len(loadings.columns)):
    ind_comp[i]=loadings[i].abs().sort_values(ascending=False).head(20)




n = len(ind_comp)

jaccard_mat = np.zeros((n, n))

for i in range(n):
    for j in range(n):

        A = set(ind_comp[i].index)
        B = set(ind_comp[j].index)

        jaccard_mat[i, j] = len(A & B) / len(A | B)

jaccard_mat = pd.DataFrame(
    jaccard_mat,
    index=[f"IC{i+1}" for i in range(n)],
    columns=[f"IC{i+1}" for i in range(n)]
)

mask = np.tril(np.ones_like(jaccard_mat, dtype=bool))
plt.figure(figsize=(20,20))
sns.heatmap(
    jaccard_mat,
    mask=mask
)
plt.savefig('Podobieństwo_niezależnych_składowych.png')
plt.close()

from sklearn.metrics import mean_squared_error
errors = []

for n in range(2, 101):

    ica = FastICA(
        n_components=n,
        random_state=42,
        max_iter=20000
    )

    # niezależne komponenty
    S = ica.fit_transform(dane_pca_scaled)

    # rekonstrukcja danych
    X_rec = S @ ica.mixing_.T

    # błąd rekonstrukcji
    mse = mean_squared_error(
        dane_pca_scaled,
        X_rec
    )

    errors.append(mse)

plt.figure(figsize=(10,5))
plt.plot(range(2,101), errors, marker='o')
plt.xlabel("Liczba komponentów ICA")
plt.ylabel("Błąd rekonstrukcji")
plt.title("ICA - błąd rekonstrukcji")
plt.grid(True)
plt.savefig("ICA_elbow.png")
plt.close()
#Ze względu na brak jednoznacznego punktu załamania wybrano 60 komponentów ICA,
# co zapewnia znaczną redukcję wymiarowości (489 → 60) przy jednoczesnym zachowaniu relatywnie niskiego błędu rekonstrukcji (0.4)


ica=FastICA(
    n_components=60,
    random_state=2026,
    max_iter=10000
)
Ica2=ica.fit_transform(dane_pca_scaled)
loadings = pd.DataFrame(
    ica.components_.T,
    index=zmienne_do_pca
)
ind_comp={}
pd.set_option('display.float_format', '{:.6f}'.format)
for i in range(len(loadings.columns)):
    ind_comp[i]=loadings[i].abs().sort_values(ascending=False).head(20)




n = len(ind_comp)

jaccard_mat = np.zeros((n, n))

for i in range(n):
    for j in range(n):

        A = set(ind_comp[i].index)
        B = set(ind_comp[j].index)

        jaccard_mat[i, j] = len(A & B) / len(A | B)

jaccard_mat = pd.DataFrame(
    jaccard_mat,
    index=[f"IC{i+1}" for i in range(n)],
    columns=[f"IC{i+1}" for i in range(n)]
)
jaccard_upper = jaccard_mat.where(
    np.triu(np.ones(jaccard_mat.shape), k=1).astype(bool)
)

mask = np.tril(np.ones_like(jaccard_mat, dtype=bool))
plt.figure(figsize=(16,16))
sns.heatmap(
    jaccard_upper,
    cmap='rocket',
    square=True,
)
plt.savefig('Podobienstwo_comp.png')
plt.close()
jaccard_upper.max().max()

for i in range(60):
    print(f"\nIC{i + 1}")
    print(
        loadings[i].sort_values(
            key=np.abs,
            ascending=False
        )
    )
print(
        loadings[0].sort_values(
            key=np.abs,
            ascending=False
        ))
IC={}
q={}
for i in range(60):
    q[i]=loadings[i].abs().quantile(0.95)
    IC[i]=loadings[i][loadings[i].abs()>q[i]]
for i in range(1,60):
    print('------')
    print(IC[i].sort_values(
        ascending=False
    ))
#https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2022.855609
#https://www.nature.com/articles/srep24968
#https://www.rndsystems.com/resources/articles/tgf-beta-induced-epithelial-mesenchymal-transition-promotes-breast-cancer-progression
#https://pmc.ncbi.nlm.nih.gov/articles/PMC11437670
#IC1 – komponent związany z sygnalizacją BMP/TGF-β oraz procesami progresji nowotworu


#IC2 Angiogeneza i mikrośrodowisko guza
#https://www.researchgate.net/publication/346268249_The_FGFFGFR_System_in_Breast_Cancer_Oncogenic_Features_and_Therapeutic_Perspectives
#https://orca.cardiff.ac.uk/id/eprint/160105/6/fonc-13-1166955.pdf
#https://www.explorationpub.com/Journals/em/Article/1001267
#https://tcr.amegroups.org/article/view/86186/html

#IC3 Regulacja epigenetyczna i supresory nowotworowe

#IC4 komponent sygnalizacji Notch i regulacji transkrypcji (Białko receptorowe rozwój komorek miesni gładkich)
#https://www.jci.org/articles/view/86114
#https://dzl.de/publication/10-1093-nar-gkac601/
#https://www.tandfonline.com/doi/abs/10.1128/mcb.22.21.7688-7700.2002


#IC5 sygnalizacji hormonalnej i metabolizmu androgenów
#https://api.repository.cam.ac.uk/server/api/core/bitstreams/2ec3efde-1075-4ba9-92f3-5ca04e60db74/content
#https://pmc.ncbi.nlm.nih.gov/articles/PMC3761226

#IC6 związany z aktywnością STAT5 i korzystnym fenotypem nowotworu


plt.figure(figsize=(15,8))
plt.subplot(3,1,1)
plt.hist(Ica2[:,0], bins=50)
plt.title('IC1')
plt.xticks(range(int(Ica2[:,0].min()), int(Ica2[:,0].max())+1,1))
plt.ylabel('Częstość')
plt.xlabel('Wartości składowej')
plt.subplot(3,1,2)
plt.hist(Ica2[:,1],bins=50)
plt.title('IC2')
plt.xticks(range(int(Ica2[:,1].min()), int(Ica2[:,1].max())+1,1))
plt.ylabel('Częstość')
plt.xlabel('Wartości składowej')
plt.subplot(3,1,3)
plt.hist(Ica2[:,2],bins=50)
plt.xticks(range(int(Ica2[:,2].min()), int(Ica2[:,2].max())+1,1))
plt.title('IC3')
plt.ylabel('Częstość')
plt.xlabel('Wartości składowej')
plt.tight_layout()
plt.savefig('histogramy_ica.png')
plt.close()
from scipy.stats import spearmanr
rho=[]
p=[]
for i in range(60):
    mask = dane2['pam50_+_claudin-low_subtype'].notna()

    r, pv = spearmanr(
        Ica2[mask, i],
        dane2.loc[mask, 'pam50_+_claudin-low_subtype']
    )
    rho.append(r)
    p.append(pv)
Komponent=[]
for i in range(60):
    K=f'IC{i+1}'
    Komponent.append(K)
Korelacje_spearman_histology=pd.DataFrame({
    'rho':rho,
    'pval':p,
    'Komponent':Komponent
})
Korelacje_spearman_histology['rho'].abs().max()



kody = pd.Categorical(
    dane2['pam50_+_claudin-low_subtype']
).codes



pca = PCA(n_components=2)
X_vis = pca.fit_transform(Ica2)

plt.figure(figsize=(15,8))
plt.scatter(
    X_vis[:,0],
    X_vis[:,1],
    c=kody,
    alpha=0.5,
    s=10
)
plt.savefig('Pca_na_ICA_.png')
plt.close()



import umap

reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.3
)

X_umap = reducer.fit_transform(Ica2)
plt.figure(figsize=(10,8))

sc = plt.scatter(
    X_umap[:,0],
    X_umap[:,1],
    c=kody,
    alpha=0.6,
    s=15
)

plt.colorbar(sc)

plt.xlabel("UMAP1")
plt.ylabel("UMAP2")

plt.savefig("UMAP_PAM50.png")
plt.close()
#Nieliniowa projekcja 60 niezależnych składowych ICA do przestrzeni dwuwymiarowej za pomocą algorytmu UMAP ujawniła częściową separację podtypów PAM50,
# co sugeruje, że komponenty ICA zawierają informacje istotne dla różnicowania molekularnych podtypów pam50_+_claudin-low_subtype