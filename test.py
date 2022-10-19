from myarray import MyArray
import numpy as np

a = np.array([[0,1,2],[2,2,2]])
b = np.array([[2,3,4],[3,3,3]])
print (id(a))
a[:] = a[:]+b[:]
print (id(a))
print(a)
