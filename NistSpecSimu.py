#本文件用于Nist谱线模拟

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r'D:\LIBS\ElementDetectation\LIBS-ElementRecogonise\Code\SpecSimuDatabase\Cr100_10000K.csv',header=0)
data=data.to_numpy()
wl=data[:,0]
intensity_sum=data[:,1]
intensity_atom=data[:,2]
intensity_ion=data[:,3]

print(intensity_atom[200])

plt.plot(wl,intensity_atom)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (a.u.)')
plt.title('NIST Spectrum Simulation of Cr')
# plt.xlim(200,900)
plt.ylim(0,1.1*max(intensity_sum))
# plt.grid()
plt.xticks(np.arange(200, 901, 100))
plt.show()