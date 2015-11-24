from numpy import *
from time import *
a=random.rand(5000,5000); t=time(); a=linalg.inv(a); time()-t
print a
