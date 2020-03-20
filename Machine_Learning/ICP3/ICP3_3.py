"Question 3"


import numpy as np
a = np.random.randint(1, 20, size=15)
print(a)
b = a.reshape(3, 5)
print(b)
c = np.where(b == np.amax(b, axis=1).reshape(-1, 1), 0, b)
print(c)