import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

df_agent_flow = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/agent_flow.csv",
                            delimiter=",", header=0)
df_raw_material = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/raw_material.csv",
                              delimiter=",", header=0)
df_reactor_sensor = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/reactor_sensor.csv",
                                delimiter=",", header=0)
df_dri_sample = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/dri_sample.csv",
                            delimiter=",", header=0)


# RAW MATERIAL
print(df_raw_material.columns)
print("First Date:", df_raw_material['Date'].iloc[0])
print("Last Date:", df_raw_material['Date'].iloc[-1])
try:
    df_raw_material['Date'] = pd.to_datetime(df_raw_material['Date'], format='%d/%m/%Y %H:%M')
except ValueError as e:
    df_raw_material['Date'] = pd.to_datetime(df_raw_material['Date'], format='mixed')

df_raw_material['Date'] = df_raw_material['Date'] - pd.Timedelta(hours=9)
df_raw_material.set_index('Date', inplace=True)
print("\n\n")


# REACTOR SENSOR
print(df_reactor_sensor.columns)
print("First Date:", df_reactor_sensor['Date'].iloc[0])
print("Last Date:", df_reactor_sensor['Date'].iloc[-1])
try:
    df_reactor_sensor['Date'] = pd.to_datetime(df_reactor_sensor['Date'], format='%d/%m/%Y %H:%M')
except ValueError as e:
    df_reactor_sensor['Date'] = pd.to_datetime(df_reactor_sensor['Date'], format='mixed')

df_reactor_sensor.set_index('Date', inplace=True)
df_reactor_sensor = df_reactor_sensor.resample('30T').agg(
    {'Temperature 1': ['mean', 'std'], 'Pressure 1': ['mean', 'std'],
     'Temperature 2': ['mean', 'std'], 'Pressure 2': ['mean', 'std'],
     'Temperature 3': ['mean', 'std'], 'Pressure 3': ['mean', 'std'],
     'Temperature 4': ['mean', 'std'], 'Pressure 4': ['mean', 'std'],
     'Temperature 5': ['mean', 'std'], 'Pressure 5': ['mean', 'std']})
df_reactor_sensor = df_reactor_sensor.rolling("9H").mean()
df_reactor_sensor = df_reactor_sensor.iloc[18:]
print(df_reactor_sensor.head())
print("\n\n")


# AGENT FLOW
print(df_agent_flow.columns)
print("First Date:", df_agent_flow['Date'].iloc[0])
print("Last Date:", df_agent_flow['Date'].iloc[-1])
try:
    df_agent_flow['Date'] = pd.to_datetime(df_agent_flow["Date"], format='%d/%m/%Y %H:%M')
except ValueError as e:
    df_agent_flow['Date'] = pd.to_datetime(df_agent_flow['Date'], format='mixed')

df_agent_flow.set_index('Date', inplace=True)
df_agent_flow = df_agent_flow.resample('30T').agg({'H2': ['mean', 'std'], 'CO': ['mean', 'std'],
                                                   'CH4': ['mean', 'std'], 'Al2O3': ['mean', 'std'],
                                                   '4SiO2': ['mean', 'std'], 'C6H10O5': ['mean', 'std'],
                                                   'C12H22O11': ['mean', 'std'], 'CaCO3': ['mean', 'std'],
                                                   'CaMgCO3': ['mean', 'std'], '3SiO2': ['mean', 'std']})
df_agent_flow = df_agent_flow.rolling("9H").mean()
df_agent_flow = df_agent_flow.iloc[18:]
print("\n\n")

final_variables = pd.concat([df_raw_material, df_reactor_sensor, df_agent_flow], axis=1, join="inner")
print("Agent flow", df_agent_flow.shape)
print("Raw material", df_raw_material.shape)
print("Reactor sensor", df_reactor_sensor.shape)
print("Final_variable", final_variables.shape)
final_variables.reset_index(inplace=True)
print(final_variables.head())
final_variables.to_csv("/Users/ochoa/PycharmProjects/Bremerhaven/generierte_p3.csv", index=False)

"""
print("DRI Sample", df_dri_sample.shape)
print(df_dri_sample.columns)

import pandas as pd

# Load the data from the CSV file
file_path = 'generierte_p3.csv'
df = pd.read_csv(file_path)
df = df.drop(columns=["Date"])

# Calculate the Pearson correlation matrix
pearson_corr_matrix = df.corr()

# Calculate the Spearman correlation matrix
spearman_corr_matrix = df.corr(method='spearman')

# Save the Pearson correlation matrix as a CSV file
pearson_output_file_path = 'pearson_correlation_matrix.csv'
pearson_corr_matrix.to_csv(pearson_output_file_path, index=True)
print(f"Pearson Correlation Matrix saved to {pearson_output_file_path}")

# Save the Spearman correlation matrix as a CSV file
spearman_output_file_path = 'spearman_correlation_matrix.csv'
spearman_corr_matrix.to_csv(spearman_output_file_path, index=True)
print(f"Spearman Correlation Matrix saved to {spearman_output_file_path}")
"""
