#numpy 
import numpy as np

print("First run:")
print(np.random.rand(3))
np.random.seed(42) # set the same seed again
print("\nSecond run (same seed):")
print(np.random.rand(3)) # same output as first run
np.random.seed(42) # set the same seed again
print("\nSecond run (same seed):")
print(np.random.rand(4)) # same output as first run
data=[1,2,3,4,]
arr=np.array(data)

data={1:2, 2:3}
arr=np.array(data.keys)
brr=np.array(data.values)
crr= np.array(data.items)

arr=np.array([[]])
print(arr.ndim)

arr=np.arrange(1,5,1)
arr=np.linspace(0,1,0.25)
# logshape finction is for printing logarethmic space logspace(start, ending-> powers, 3->numbers of elenments)
arr3 = np.logspace(0, 2, 3)
#matrix
arr=np.ones([2,3])
arr=np.zeros([3,2])
#full
arr=np.full((3,4),5)
arr=np.full((4), 7)
arr=np.empty([2,3], dtype=int)
#rand, randn, randint
arr=np.rand((2,3)) #between -0 and one 2 rows and 3 columns
arr=np.randn((3,2))# normal distribution with mean=0 and standard deviation =one
arr=np.randint(1, 100, (3,2))#between 1 and 100 and 3 rows and 2 columns

data=[1,2,3,4,5,6,7,8,9]
arr=np.array(data)
arr=np.reshape([3,2])
arr=np.reshape((3,3))

#ravel function is used to create a multi dim arrray into 1 dimension
'''ravel function creates a view of the original array, so any changes made
to the raveled array will affect the original array'''
arr=np.array([[1,2,3], [4,5,6]])
arr=np.ravel()
print(arr)

#to counter this flatten() function is used
arr=np.flatten()
'''
it creates a copy of the original array so that changes made will not be affected 
in the original matrix'''

#take(arr, index)
arr=np.array([[1,2,3,4], [4,5,6,7]])
index=[0,2]
print(np.take(arr, index))

np.concatinate(arr, brr)
np.vstack(arr, brr)
np.hstack(arr, brr)


arr=np.array([1,2,np.inf,-3, -np.inf])
np.isinf(arr)  # checking for infinite values
np.nan_to_num(arr, posinf=100, neginf=-100)  # replace inf with 100 and -inf with -100
arr
arr.astype(int)