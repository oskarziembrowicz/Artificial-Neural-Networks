import numpy as np

data_in = np.array([[0,0],[0,1],[1,0],[1,1]])
# data_in = np.array([0,1,2,3])
expected_out = np.array([1,0,1,0])
wages = np.array([0.1,0.1,0.1,0.1])
bias = 0
learn_factor = 0.2

max_it = 100
for i in range(max_it):
    multiplyed_input = (data_in.T * wages).T
    output = [1 if x + bias > 0 else 0 for x in multiplyed_input]
    diff = np.subtract(expected_out, output)
    correction = wages + diff * data_in * learn_factor
    wages = correction
    if all(0 == x for x in diff):
        break


print(wages)
print(output)
