import numpy as np
import scipy
import scipy.stats

speed = [86,87,88,86,87,85,86]

nums = [12,34,17,5,6,7,8,5,4,6,88,9]

# getting the average of a list with numpy
mean = np.mean(nums)

# getting the middle  number in the list
med = np.median(nums)

# getting the most occuring number

mode = scipy.stats.mode(nums)

#getting the standard deviation
x = np.std(speed)

print(mode)

