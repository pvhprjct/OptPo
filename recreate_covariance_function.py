# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 17:10:18 2021

@author: pvghd
"""
import numpy as np
import pandas as pd
from datetime import date
import pandas_datareader as pdr
import matplotlib.pyplot as plt


# How does the portfolio.cov() function work?

portfolio = pd.DataFrame( [(1,3), (2,5), (3,7), (4,2), (5,4), (6,1)], 
                         columns=['a', 'b'] )

covariance = portfolio.cov()
print( covariance )

# Now let us recreate the function:

a = [1,2,3,4,5,6]
b = [3,5,7,2,4,1]

"""
To calculate the pairwise covariance. You take the mean value of the array
and subtract it from each of the components. Then you multiply both arrays
component by component and sum them. You divide everything by the number
of components of the arrays (which must have the same dimension) minus one.
The minus one comes from the Bessel correction.
Just like in http://nambis.bplaced.net/nambis/SMDV/SMDV_Kapitel7.pdf p.7
(with corrections)
"""

def covariance_func(a,b):
    a_2 = a - np.mean(a)
    b_2 = b - np.mean(b)
    return np.sum( a_2 * b_2 ) /  (len(a)-1)

print()
print('variance a =', covariance_func(a, a) )
print('covariance ab =', covariance_func(a, b) )
print('covariance ab =', covariance_func(b, a) )
print('variance b =', covariance_func(b, b) )

a_2 = a - np.mean(a)
b_2 = b - np.mean(b)
