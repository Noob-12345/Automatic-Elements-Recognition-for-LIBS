import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

kB=8.617330350e-5 #eV/K


def Boltzmann_fit(I,wl, A, g, E):
    y = np.log(I*wl/ (g * A))
    coefficients = np.polyfit(E, y, 1) 
    slope, intercept = coefficients
    T = -1/(slope * kB) 
    
    # 画图
    plt.figure(figsize=(8, 6))
    plt.scatter(E, y, color='blue', label='Data points')
    plt.plot(E, slope * E + intercept, color='red', label='Fitted line')
    plt.xlabel('Energy (eV)')
    plt.ylabel('ln(I*wl/g*A)')
    plt.title('Boltzmann Plot T={:.2f} K'.format(T))
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return coefficients, slope, intercept, T, y

indices_to_keep = [0,1, 2, 3, 4,5,6,7,8]  # 要保留的索引

A = np.array([1.48e+08, 1.50e+08, 1.62e+08, 3.15e+07,3.07e+07,3.16e+07,5.09e+07,5.14e+07,5.06e+07])[indices_to_keep]
g = np.array([9,7,5,9,7,5,3,5,7])[indices_to_keep]
E = np.array([3.463528,3.449264,3.437934,2.913481,2.899536,2.889452,3.323012,3.322313,3.321222])[indices_to_keep] #eV
wl = np.array([357.869,359.347,360.534,425.433, 427.482, 428.97,520.453,520.606,520.845])[indices_to_keep]*10**(-9)
I = np.array([3010,2403,1859,1021,779,577,279,470,640])[indices_to_keep]

coefficients, slope, intercept, T, y = Boltzmann_fit(I,wl, A, g, E)


print(f"Fitted Temperature: {T} K")