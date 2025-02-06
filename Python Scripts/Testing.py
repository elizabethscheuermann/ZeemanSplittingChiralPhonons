import Base
import FourLevel_Base
import matplotlib.pyplot as plt
import numpy as np

T = 1
x = [a for (a, b) in np.linalg.eig(Base.H_MF_tri([1, 0], 1, 0))]
print(x)