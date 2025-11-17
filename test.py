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

indices_to_keep = [0,1,2,3,4,5,6,7]  # 要保留的索引

A = np.array([0.0281,0.0281,0.101,0.121,0.616,0.614,0.429,0.514])[indices_to_keep]
g = np.array([4,2,4,6,4,2,4,6])[indices_to_keep]
E = np.array([3.75,3.75,4.28,4.28,2.10,2.10,3.61,3.616])[indices_to_keep] #eV
wl = np.array([330.24,330.30,568.2,568.8,588.9,589.5,818.3,819.5])[indices_to_keep]*10**(-9)
I = np.array([186,92,215,424,15929,7950,1375,2701])[indices_to_keep]

coefficients, slope, intercept, T, y = Boltzmann_fit(I,wl, A, g, E)


print(f"Fitted Temperature: {T} K")