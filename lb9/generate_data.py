import pandas as pd
import numpy as np

# Setting up data
number_of_normal_rows = 1000
number_of_abnormal_rows = 50
number_of_traits = 10

# Generating data
normal_data = np.random.normal(0,1,(number_of_normal_rows, number_of_traits))
abnormal_data = np.random.normal(5,1.5, (number_of_abnormal_rows, number_of_traits))

# Joining data
data = np.vstack((normal_data, abnormal_data))

#Create labels (0 - normal, 1 - abnormal)
labels = np.concatenate([np.zeros(number_of_normal_rows), np.ones(number_of_abnormal_rows)])

df = pd.DataFrame(data, columns=[f'trait_{i+1}' for i in range(number_of_traits)])
df['anomaly'] = labels

# Save to file
df.to_csv('data_for_autoencoder.csv', index=False)

print("Data was saved in file: 'data_for_autoencoder.csv'")