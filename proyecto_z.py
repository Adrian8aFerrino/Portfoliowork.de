import time
import simpy
import numpy as np
import pandas as pd
from itertools import count
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data_augmented_mensa = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/sim_mensa.csv", header=None)
data_augmented_arrival = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/sim_arrival.csv", header=None)
data_augmented_cashier = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/sim_cashier.csv", header=None)
data_augmented_mensa.drop(data_augmented_mensa.columns[0], axis=1, inplace=True)
data_augmented_arrival.drop(data_augmented_arrival.columns[0], axis=1, inplace=True)
data_augmented_cashier.drop(data_augmented_cashier.columns[0], axis=1, inplace=True)
size_mensa = data_augmented_mensa.shape[0]
size_arrival = data_augmented_arrival.shape[0]
size_cashier = data_augmented_cashier.shape[0]
print("Original data size ARRIVAL: ", size_arrival)
print("Original data size MENSA: ", size_mensa)
print("Original data size CASHIER: ", size_cashier)


def augment_exp(data_augmented, sim_iteration, correction_factor, label):
    mean = data_augmented[1].mean()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    data_augmented[1].plot(kind="hist", alpha=0.5, label="Mensa", color="green", bins=100)
    plt.xlabel(f"{label} (seconds)")
    plt.ylabel("Frequency")
    plt.title("Population Distribution Original")
    samples = np.random.exponential(scale=mean, size=sim_iteration)
    new_data = pd.DataFrame({1: samples})
    data_augmented = pd.concat([data_augmented, new_data], ignore_index=True)
    data_augmented[1] = data_augmented[1] + correction_factor
    size_two = data_augmented.shape[0]
    plt.subplot(1, 2, 2)
    data_augmented[1].plot(kind="hist", alpha=0.5, label="Mensa", color="orange", bins=100)
    plt.xlabel(f"{label} (seconds)")
    plt.ylabel("Frequency")
    plt.title("Population Distribution Generated")
    plt.tight_layout()
    # plt.show()
    return data_augmented[1], size_two


def augment_poisson(data_augmented, sim_iteration, label):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    data_augmented[1].plot(kind="hist", alpha=0.5, label="Mensa", color="green", bins=100)
    plt.xlabel(f"{label} (People per minute)")
    plt.ylabel("Frequency")
    plt.title("Population Distribution Original")
    mean = data_augmented[1].mean()
    samples = np.random.poisson(mean, size=sim_iteration)
    new_data = pd.DataFrame({1: samples})
    data_augmented = pd.concat([data_augmented, new_data], ignore_index=True)
    size_two = data_augmented.shape[0]
    plt.subplot(1, 2, 2)
    data_augmented[1].plot(kind="hist", alpha=0.5, label="Mensa", color="orange", bins=100)
    plt.xlabel(f"{label} (People per minute)")
    plt.ylabel("Frequency")
    plt.title("Population Distribution Generated")
    plt.tight_layout()
    # plt.show()
    return data_augmented[1], size_two


new_arrival, result_augmentation_arrival = augment_poisson(data_augmented=data_augmented_arrival, sim_iteration=404,
                                                           label="Arrival rate")
# new_arrival.to_csv('generierte_ankunftsrate.csv', index=False, header=False)
print("\nAugmented data size ARRIVAL: ", result_augmentation_arrival)

new_mensa, result_augmentation_mensa = augment_exp(data_augmented=data_augmented_mensa, sim_iteration=632,
                                                   correction_factor=8.54, label="Mensa service time")
# new_mensa.to_csv('generierte_mensadienstzeit.csv', index=False, header=False)
print("Augmented data size MENSA: ", result_augmentation_mensa)

new_cashier, result_augmentation_cashier = augment_exp(data_augmented=data_augmented_cashier, sim_iteration=636,
                                                       correction_factor=4.05, label="Cashier service time")
# new_cashier.to_csv('generierte_cashierdienstzeit.csv', index=False, header=False)
print("Augmented data size CASHIER: ", result_augmentation_cashier)


avg_arrival_rate = np.mean(new_arrival) / 60
avg_service_time_uno = new_mensa.mean()
avg_service_time_dos = new_cashier.mean()

capacity_mensa = 3
capacity_cashier = 2
utilization_factor_mensa = (avg_arrival_rate * avg_service_time_uno) / capacity_mensa
utilization_factor_cashier = (avg_arrival_rate * avg_service_time_dos) / capacity_cashier

print("\nUtilization factor of the MENSA station:", utilization_factor_mensa)
print("Utilization factor of the CASHIER station:", utilization_factor_cashier)

print("\n\n - - - - - -  - - - - - -  - - - - - - INITIATING SIMULATION - - - - - -  - - - - - -  - - - - - - ")

env = simpy.Environment()
server_number_uno = simpy.Resource(env, capacity=capacity_mensa)
server_number_dos = simpy.Resource(env, capacity=capacity_cashier)

station_mensa = simpy.Store(env)
queue_between = simpy.Store(env)
station_cashier = simpy.Store(env)
client_data = pd.DataFrame(columns=['Client Number', 'Arrival Rate', 'Waiting time before Mensa', 'Service Time Mensa',
                                    "Waiting time before Cashier", "Service Time Cashier", "Lead Time"])


def arrival_process_stations(env, client_number, client_data):
    while True:
        arrival_select = np.random.choice(new_arrival.values) * 1.72
        client_data.loc[client_number, 'Arrival Rate'] = arrival_select
        if arrival_select != 0:
            interarrival_time = 60 / arrival_select
        else:
            interarrival_time = 60
        yield env.timeout(interarrival_time)
        client_number += 1
        station_mensa.put((env.now, client_number))
        env.process(service_process_mensa(env, client_data))


def service_process_mensa(env, client_data):
    while True:
        with server_number_uno.request() as request:
            yield request
            arrival_time, client_number = yield station_mensa.get()
            service_time_uno = np.random.choice(new_mensa.values)
            yield env.timeout(service_time_uno)
            station_cashier.put((env.now, client_number))
            client_data.loc[client_number, "Client Number"] = client_number
            client_data.loc[client_number, "Service Time Mensa"] = service_time_uno
            client_data.loc[client_number, 'Waiting time before Mensa'] = env.now - arrival_time
            env.process(service_process_cashier(env, client_data))


def service_process_cashier(env, client_data):
    while True:
        with server_number_dos.request() as request:
            yield request
            arrival_time, client_number = yield station_cashier.get()
            service_time_dos = np.random.choice(new_cashier.values)
            yield env.timeout(service_time_dos)
            client_data.loc[client_number, "Service Time Cashier"] = service_time_dos
            client_data.loc[client_number, 'Waiting time before Cashier'] = env.now - arrival_time


env.process(arrival_process_stations(env, client_number=0, client_data=client_data))
work_days = 3 * 20
env.run(until=(3600 * work_days))

utilization_factor1 = ((client_data["Arrival Rate"].mean() / 60) * client_data[
    "Service Time Mensa"].mean()) / capacity_mensa
utilization_factor2 = ((client_data["Arrival Rate"].mean() / 60) * client_data["Service Time Cashier"].mean()) / \
                      capacity_cashier
client_data["Waiting time before Mensa"] = (client_data["Waiting time before Mensa"] -
                                            client_data["Service Time Mensa"]).abs()
client_data["Waiting time before Cashier"] = (client_data["Waiting time before Cashier"] -
                                              client_data["Service Time Cashier"]).abs()
client_data["Lead Time"] = client_data[['Waiting time before Mensa', 'Service Time Mensa',
                                        "Waiting time before Cashier", "Service Time Cashier", "Lead Time"]].sum(axis=1)
client_data = client_data.drop('Arrival Rate', axis=1)

num_rows = 5000
counter = 0
for _, row in client_data.iterrows():
    print("\n\n", row)
    counter += 1
    time.sleep(1)
    if counter == num_rows:
        break

print("\n\n - - - - - -  - - - - - -  - - - - - - END OF SIMULATION - - - - - -  - - - - - -  - - - - - - ")
print("\n\nUtilization factor of the MENSA station:", utilization_factor1)
print("Utilization factor of the CASHIER station:", utilization_factor2)
print("Total number of customers: ", client_data.shape[0])
print("\n\n", client_data.mean())

x_axis = []
y_axis_uno = []
y_axis_dos = []
fig, ax = plt.subplots()
ax.plot(x_axis, y_axis_uno)
ax.plot(x_axis, y_axis_dos)
counter_graph = count(0, 1)


def update_graph(i):
    index_count = next(counter_graph)
    x_axis.append(client_data["Client Number"].iloc[index_count])
    y_axis_uno.append(client_data["Waiting time before Mensa"].iloc[index_count])
    y_axis_dos.append(client_data["Waiting time before Cashier"].iloc[index_count])
    plt.cla()
    ax.plot(x_axis, y_axis_uno, color="red", label="Waiting time in the MENSA station")
    ax.plot(x_axis, y_axis_dos, color="blue", label="Waiting time in the CASHIER station")
    plt.xlabel("Client Number")
    plt.ylabel("Waiting time (seconds)")
    plt.title("Waiting time in both stations")
    ax.legend(loc="upper left")


animation = FuncAnimation(fig=fig, func=update_graph, interval=200, save_count=100)
# plt.show()
