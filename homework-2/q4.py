#q4
#for 100 points in lagrange
import numpy as np
from scipy.interpolate import lagrange
from sklearn.metrics import mean_squared_error
from numpy.polynomial.polynomial import Polynomial

a, b, n = 0, 2*np.pi, 100

x_train = np.random.uniform(a, b, n)
y_train = np.sin(x_train)
model = lagrange(x_train, y_train)

x_test = np.random.uniform(a, b, 25)
y_test = np.sin(x_test)

y_train_pred = model(x_train)
y_test_pred = model(x_test)

train_error = mean_squared_error(y_train, y_train_pred)
test_error = mean_squared_error(y_test, y_test_pred)

print(f'Training error:(log) {np.log10(train_error)}')
print(f'Testing error: (log) {np.log10(test_error)}')

train_errors=[]
test_errors=[]
for std_dev in [0.1, 0.3, 0.5, 0.7, 1.0]:
    xn_train = x_train + np.random.normal(0, std_dev, n)
    yn_train = np.sin(xn_train)

    nlagrange = lagrange(xn_train, yn_train)

    yn_train_pred = nlagrange(xn_train)
    yn_test_pred = nlagrange(x_test)

    ntrain_error = mean_squared_error(yn_train, yn_train_pred)
    ntest_error = mean_squared_error(y_test, yn_test_pred)
    train_errors.append(np.log10(ntrain_error))
    test_errors.append(ntest_error)
    print(f"\nTrain Error (log) (Std Dev {std_dev}): {np.log10(ntrain_error)}")
    print(f"Test Error (log)(Std Dev {std_dev}): {np.log10(ntest_error)}")
plt.figure(figsize=(10, 5))
plt.plot([0.1,0.3,0.5,0.7,1.0],train_errors , marker='o')
plt.title("Learning Curve")
plt.xlabel("std_dev")
plt.ylabel("Test Error")
plt.show()
