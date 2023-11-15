import time
import warnings
import datetime
import pulp as plp
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import skew, kurtosis, jarque_bera
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.6f' % x)

staff_df = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/staff_elements.csv", encoding='latin1',
                       header=0)
store_df = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/store_elements.csv", encoding='latin1',
                       header=0, dtype={'Beleggröße': str, 'Produktverkäufe': str})

staff_df = staff_df.replace('k.A', np.nan)
staff_df = staff_df.fillna('31/03/2023')
dum1 = staff_df.Schichtverfügbarkeit.str.get_dummies()
dum2 = staff_df.Qualifikationen.str.get_dummies()
staff_df[['Überstunden', 'Schichtpräferenz']] = staff_df[['Überstunden', 'Schichtpräferenz']].replace(
    {'JA': 1, 'NEIN': 0}).astype('bool')
staff_df['Einstellungsdatum'] = pd.to_datetime(staff_df['Einstellungsdatum'], format='%d/%m/%Y')
staff_df['Entlassungsdatum'] = pd.to_datetime(staff_df['Entlassungsdatum'], format='%d/%m/%Y')
staff_df['Tage_zwischen'] = (staff_df['Entlassungsdatum'] - staff_df['Einstellungsdatum']).dt.days
staff_df['Gemischtwarenladen'] = staff_df['Gemischtwarenladen'].astype(str)
staff_df['Überstunden'] = staff_df['Überstunden'].astype(int)
staff_df['Schichtpräferenz'] = staff_df['Schichtpräferenz'].astype(int)
dum3 = staff_df.Gemischtwarenladen.str.get_dummies()
# print("\nStaff data types:\n", staff_df.dtypes)

store_df['Datum'] = pd.to_datetime(store_df['Datum'], dayfirst=True)
store_df['Tag_des_Jahres'] = store_df['Datum'].dt.dayofyear
store_df['Gemischtwarenladen'] = store_df['Gemischtwarenladen'].astype(str)
store_df[['Feiertag', 'Sonderaktionen']] = store_df[['Feiertag', 'Sonderaktionen']].replace(
    {'JA': 1, 'NEIN': 0}).astype('bool')
store_df['Beleggröße'] = store_df['Beleggröße'].str.replace(',', '').str.replace('', '').astype('float')
store_df['Produktverkäufe'] = store_df['Produktverkäufe'].str.replace(',', '').str.replace('', '').astype('float')
# print("\nStore data types\n", store_df.dtypes)


def custom_describe(dataframe):
    description = dataframe.describe()
    description.loc['skewness'] = skew(dataframe)
    description.loc['kurtosis'] = kurtosis(dataframe)
    description.loc['jarque-bera'] = jarque_bera(dataframe)[1]
    return description


# Descriptive statistics (Measures of central tendency, measures of variability)
new_store_df = store_df[store_df['Produktverkäufe'] != 0]
# print(custom_describe(store_df['Beleggröße']))
# print(custom_describe(new_store_df['Beleggröße']))


# Descriptive statistics (Measures of frequency)
gs_eins = gridspec.GridSpec(2, 2)
fig_eins = plt.figure()
ax1_eins = fig_eins.add_subplot(gs_eins[0, 0])
ax1_eins.plot([0, 1])
ax2_eins = fig_eins.add_subplot(gs_eins[0, 1])
ax2_eins.plot([0, 1])
ax3_eins = fig_eins.add_subplot(gs_eins[1, 0])
ax3_eins.plot([0, 1])
ax4_eins = fig_eins.add_subplot(gs_eins[1, 1])
ax4_eins.plot([1, 1])

fig_eins.suptitle('Häufigkeiten (Personal)')
staff_count1 = staff_df['Schichtverfügbarkeit'].value_counts()
staff_count1.plot(kind='bar', ax=ax1_eins, color='navy')
ax1_eins.tick_params(labelrotation=0)
ax1_eins.set_title('Schichtverfügbarkeit')
for rect in ax1_eins.patches:
    height = rect.get_height()
    ax1_eins.text(rect.get_x() + rect.get_width() / 2, height, int(height), ha='center', va='bottom', fontweight='bold')

staff_count2 = staff_df['Qualifikationen'].value_counts()
staff_count2.plot(kind='bar', ax=ax2_eins, color='darkred')
ax2_eins.tick_params(labelrotation=0)
ax2_eins.set_title('Qualifikationen')
for rect in ax2_eins.patches:
    height = rect.get_height()
    ax2_eins.text(rect.get_x() + rect.get_width() / 2, height, int(height), ha='center', va='bottom', fontweight='bold')

staff_count3 = staff_df['Gemischtwarenladen'].value_counts()
staff_count3.plot(kind='bar', ax=ax3_eins, color='orange')
ax3_eins.tick_params(labelrotation=0)
ax3_eins.set_title('Gemischtwarenladen')
for rect in ax3_eins.patches:
    height = rect.get_height()
    ax3_eins.text(rect.get_x() + rect.get_width() / 2, height, int(height), ha='center', va='bottom', fontweight='bold')

staff_count4 = staff_df['Schichtpräferenz'].value_counts()
staff_count4.plot(kind='bar', ax=ax4_eins, color='green')
ax4_eins.tick_params(labelrotation=0)
ax4_eins.set_title('Schichtpräferenz')
for rect in ax4_eins.patches:
    height = rect.get_height()
    ax4_eins.text(rect.get_x() + rect.get_width() / 2, height, int(height), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
# plt.show()


# Descriptive statistics (Measures of distribution)
gs_zwei = gridspec.GridSpec(2, 2)
fig_zwei = plt.figure()
ax1_zwei = fig_zwei.add_subplot(gs_zwei[0, 0])
ax1_zwei.plot([0, 1])
ax2_zwei = fig_zwei.add_subplot(gs_zwei[0, 1])
ax2_zwei.plot([0, 1])
ax3_zwei = fig_zwei.add_subplot(gs_zwei[1, 0])
ax3_zwei.plot([0, 1])
ax4_zwei = fig_zwei.add_subplot(gs_zwei[1, 1])
ax4_zwei.plot([1, 1])

fig_zwei.suptitle('Maße der Häufigkeitsverteilung')
store_hist1 = new_store_df['Kundenfrequenz']
store_hist1.plot(kind='hist', ax=ax1_zwei, color='darkcyan', bins=30)
ax1_zwei.tick_params(labelrotation=0)
ax1_zwei.set_title('Kundenfrequenz')

store_hist2 = new_store_df['Beleggröße']
store_hist2.plot(kind='hist', ax=ax2_zwei, color='firebrick', bins=30)
ax2_zwei.tick_params(labelrotation=0)
ax2_zwei.set_title('Beleggröße')

store_hist3 = new_store_df['Produktverkäufe']
store_hist3.plot(kind='hist', ax=ax3_zwei, color='orangered', bins=30)
ax3_zwei.tick_params(labelrotation=0)
ax3_zwei.set_title('Produktverkäufe')

store_hist4 = staff_df['Tage_zwischen']
store_hist4.plot(kind='hist', ax=ax4_zwei, color='limegreen', bins=30)
ax4_zwei.tick_params(labelrotation=0)
ax4_zwei.set_title('Arbeitstagen')
print(custom_describe(staff_df['Tage_zwischen']))

plt.tight_layout()
# plt.show()


# Descriptive statistics (Correlation matrix)
staff_df_new = staff_df.drop(['ID Nummer', 'Vorname', 'Nachname', 'Schichtverfügbarkeit', 'Qualifikationen',
                              'Gemischtwarenladen'], axis=1)
final_staff = pd.concat([staff_df_new, dum1, dum2, dum3], axis=1)
corr_matrix = final_staff.corr()
# corr_matrix.to_csv('corr_matrix_v2.csv', index=True, header=True)


def plot_line_graphs(dataframe):
    dataframe['Jahr'] = dataframe['Einstellungsdatum'].dt.year
    dataframe['Monat'] = dataframe['Einstellungsdatum'].dt.month
    pivot_tabelle = pd.pivot_table(dataframe, values='Tage_zwischen', index='Monat', columns='Jahr', aggfunc='mean')

    fig, nx = plt.subplots()
    for jahr in pivot_tabelle.columns:
        nx.plot(pivot_tabelle.index, pivot_tabelle[jahr], label=jahr, linestyle='--', marker='8')
        nx.grid(True, which='both')

    nx.set_xlabel('Monaten')
    nx.set_ylabel('Durchschnittliche Arbeitstage')
    nx.set_title('Vergleich der Arbeitstagdaten pro Jahr')
    nx.legend()
    # plt.show()


grouped_store = dict(tuple(new_store_df.groupby('Gemischtwarenladen')))
grouped_staff = dict(tuple(staff_df.groupby('Gemischtwarenladen')))

group_uno = grouped_staff["1"]
group_dos = grouped_staff["2"]
group_tres = grouped_staff["3"]
group_cuatro = grouped_staff["4"]
group_cinco = grouped_staff["5"]
group_seis = grouped_staff["6"]
group_siete = grouped_staff["7"]
group_ocho = grouped_staff["8"]

plot_line_graphs(group_cuatro)
plot_line_graphs(group_ocho)

# PART 2
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

pd.set_option('display.max_columns', None)

staff_df = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/staff_elements.csv", encoding='latin1',
                       header=0)
staff_df = staff_df.replace('k.A', np.nan)
staff_df = staff_df.fillna('31/03/2023')
staff_df[['Überstunden', 'Schichtpräferenz']] = staff_df[['Überstunden', 'Schichtpräferenz']].replace(
    {'JA': 1, 'NEIN': 0}).astype('bool')
staff_df['Einstellungsdatum'] = pd.to_datetime(staff_df['Einstellungsdatum'], format='%d/%m/%Y')
staff_df['Entlassungsdatum'] = pd.to_datetime(staff_df['Entlassungsdatum'], format='%d/%m/%Y')
staff_df['Tage_zwischen'] = (staff_df['Entlassungsdatum'] - staff_df['Einstellungsdatum']).dt.days
staff_df['Gemischtwarenladen'] = staff_df['Gemischtwarenladen'].astype(str)
staff_df['Überstunden'] = staff_df['Überstunden'].astype(int)
staff_df['Schichtpräferenz'] = staff_df['Schichtpräferenz'].astype(int)
print("\nStaff data types:\n", staff_df.dtypes)

store_df = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/store_elements.csv", encoding='latin1',
                       header=0, dtype={'Beleggröße': str, 'Produktverkäufe': str})
store_df['Datum'] = pd.to_datetime(store_df['Datum'], dayfirst=True)
store_df['Tag_des_Jahres'] = store_df['Datum'].dt.dayofyear
store_df['Gemischtwarenladen'] = store_df['Gemischtwarenladen'].astype(str)
store_df[['Feiertag', 'Sonderaktionen']] = store_df[['Feiertag', 'Sonderaktionen']].replace(
    {'JA': 1, 'NEIN': 0}).astype('bool')
store_df['Beleggröße'] = store_df['Beleggröße'].str.replace(',', '').str.replace('', '').astype('float')
store_df['Produktverkäufe'] = store_df['Produktverkäufe'].str.replace(',', '').str.replace('', '').astype('float')
print("\nStore data types:\n", store_df.dtypes)

date_filter = store_df[store_df.Datum.dt.weekday == 6]
date_filter = date_filter[date_filter["Feiertag"] != 1]
date_list = list(date_filter["Datum"])
date_list_str = [date.strftime('%Y-%m-%d %H:%M:%S') for date in date_list]
date_list = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').strftime('%d.%m.%Y') for date in date_list_str]
staff_df['date_range'] = staff_df.apply(lambda row: pd.date_range(start=row['Einstellungsdatum'],
                                                                  end=row['Entlassungsdatum'], freq='D'), axis=1)
datumsbereich = pd.date_range(start="01/01/2015", end="31/03/2023", freq='D')
for date in datumsbereich:
    staff_df[date.strftime('%d.%m.%Y')] = staff_df['date_range'].apply(lambda x: 1 if date in x else 0)

staff_df.drop('date_range', axis=1, inplace=True)
print("Datenrahmenform für alle Mitarbeiter", staff_df.shape)
staff_df = staff_df.drop(columns=date_list)
print("Datenrahmenform für alle Mitarbeiter nach Herausfiltern von Daten:", staff_df.shape)
grouped_staff = dict(tuple(staff_df.groupby('Gemischtwarenladen')))
staff_df = grouped_staff["1"]
print("Datenrahmenform für alle Mitarbeiter in Gemischtwarenladen 1:", staff_df.shape)

start = time.time()
prob = plp.LpProblem("Employee Scheduling", plp.LpMinimize)
personal = staff_df['ID Nummer'].tolist()
schichtverfugbarkeit = staff_df['Schichtverfügbarkeit'].unique().tolist()
qualifikationen = staff_df['Qualifikationen'].unique().tolist()
datum = datumsbereich.strftime('%d.%m.%Y').tolist()
datum = [x for x in datum if x not in date_list]

x = plp.LpVariable.dicts("x", [(e, s, d) for e in personal for s in schichtverfugbarkeit for d in datum], cat='Binary')
prob += plp.lpSum([x[(e, s, d)] for e in personal for s in schichtverfugbarkeit for d in datum])

for d in datum:
    for s in schichtverfugbarkeit:
        prob += plp.lpSum([x[(e, s, d)] for e in personal]) >= staff_df[staff_df['Schichtverfügbarkeit'] == s][d].sum()
        for r in qualifikationen:
            prob += plp.lpSum([x[(e, s, d)] for e in personal if staff_df.loc[staff_df['ID Nummer'] == e
            , 'Qualifikationen'].item() == r]) >= \
                    staff_df[(staff_df['Schichtverfügbarkeit'] == s) & (staff_df[d] == 1) &
                             (staff_df['Qualifikationen'] == r)][d].sum()
            print(f"Tag: {d} -- Schichtverfügbarkeit: {s} -- Qualifikationen: {r}")

for e in personal:
    for i in range(len(datum) - 4):
        prob += plp.lpSum([x[(e, s, datum[j])] for j in range(i, i + 5) for s in schichtverfugbarkeit]) <= 5

prob.solve()
end = time.time()
print("End time: ", end - start)

print("Status:", plp.LpStatus[prob.status])
results = pd.DataFrame(columns=["Variable", "Value", ""])
results_list = []
for r4 in prob.variables():
    results_list.append({"Variable": r4.name, "Value": r4.varValue})
results = pd.DataFrame(results_list)
print("Total Cost =", plp.value(prob.objective))
results.to_csv("results.csv", index=False)

# PART 3
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

pd.set_option('display.max_columns', None)

staff_df = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/staff_elements.csv", encoding='latin1',
                       header=0)
staff_df = staff_df.replace('k.A', np.nan)
staff_df = staff_df.fillna('31/03/2023')
staff_df[['Überstunden', 'Schichtpräferenz']] = staff_df[['Überstunden', 'Schichtpräferenz']].replace(
    {'JA': 1, 'NEIN': 0}).astype('bool')
staff_df['Einstellungsdatum'] = pd.to_datetime(staff_df['Einstellungsdatum'], format='%d/%m/%Y')
staff_df['Entlassungsdatum'] = pd.to_datetime(staff_df['Entlassungsdatum'], format='%d/%m/%Y')
staff_df['Tage_zwischen'] = (staff_df['Entlassungsdatum'] - staff_df['Einstellungsdatum']).dt.days
staff_df['Gemischtwarenladen'] = staff_df['Gemischtwarenladen'].astype(str)
staff_df['Überstunden'] = staff_df['Überstunden'].astype(int)
staff_df['Schichtpräferenz'] = staff_df['Schichtpräferenz'].astype(int)
print("\nStaff data types:\n", staff_df.dtypes)

store_df = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/store_elements.csv", encoding='latin1',
                       header=0, dtype={'Beleggröße': str, 'Produktverkäufe': str})
store_df['Datum'] = pd.to_datetime(store_df['Datum'], dayfirst=True)
store_df['Tag_des_Jahres'] = store_df['Datum'].dt.dayofyear
store_df['Gemischtwarenladen'] = store_df['Gemischtwarenladen'].astype(str)
store_df[['Feiertag', 'Sonderaktionen']] = store_df[['Feiertag', 'Sonderaktionen']].replace(
    {'JA': 1, 'NEIN': 0}).astype('bool')
store_df['Beleggröße'] = store_df['Beleggröße'].str.replace(',', '').str.replace('', '').astype('float')
store_df['Produktverkäufe'] = store_df['Produktverkäufe'].str.replace(',', '').str.replace('', '').astype('float')
print("\nStore data types:\n", store_df.dtypes)


print("Datenrahmenform für alle Mitarbeiter:", staff_df.shape)
date_filter = store_df[store_df.Datum.dt.weekday == 6]
date_filter = date_filter[date_filter["Feiertag"] != 1]
date_list = list(date_filter["Datum"])
date_list_str = [date.strftime('%Y-%m-%d %H:%M:%S') for date in date_list]
date_list = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').strftime('%d.%m.%Y') for date in date_list_str]

staff_df['date_range'] = staff_df.apply(lambda row: pd.date_range(start=row['Einstellungsdatum'],
                                                                  end=row['Entlassungsdatum'], freq='D'), axis=1)
datumsbereich = pd.date_range(start="01/01/2015", end="31/03/2023", freq='D')
for date in datumsbereich:
    staff_df[date.strftime('%d.%m.%Y')] = staff_df['date_range'].apply(lambda x: 1 if date in x else 0)

staff_df.drop('date_range', axis=1, inplace=True)
print("Datenrahmenform für alle Mitarbeiter:", staff_df.shape)
staff_df = staff_df.drop(columns=date_list)
print("Datenrahmenform für alle Mitarbeiter:", staff_df.shape)

# LOGISTICS REGRESSION (LOGIT FUNCTION)
dumm_eins = pd.get_dummies(staff_df["Schichtverfügbarkeit"], prefix="Schichtverfügbarkeit")
dumm_zwei = pd.get_dummies(staff_df["Qualifikationen"], prefix="Qualifikationen")
dumm_drei = pd.get_dummies(staff_df["Gemischtwarenladen"], prefix="Gemischtwarenladen")

df_logit = pd.concat([staff_df, dumm_eins, dumm_zwei, dumm_drei], axis=1)
predictors = ["Überstunden", "Schichtpräferenz", "Schichtverfügbarkeit_Morgenschicht",
              "Schichtverfügbarkeit_Nachmittagsschicht", "Qualifikationen_Aushilfe", "Qualifikationen_Reinigungskraft",
              "Qualifikationen_Verkäufer", "Gemischtwarenladen_1", "Gemischtwarenladen_2", "Gemischtwarenladen_3",
              "Gemischtwarenladen_4", "Gemischtwarenladen_5", "Gemischtwarenladen_6", "Gemischtwarenladen_7",
              "Gemischtwarenladen_8"]
mean_resp = df_logit["Tage_zwischen"].mean()
df_logit["response"] = (df_logit["Tage_zwischen"] < mean_resp).astype(int)
print("\n\nDF_LOGIT data types:\n", df_logit.dtypes)
response = ["response"]

X_train, X_test, y_train, y_test = train_test_split(df_logit[predictors], df_logit[response], train_size=0.8,
                                                    random_state=0)
model = sm.Logit(y_train, X_train).fit()
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print("\n\n\n", model.summary())
