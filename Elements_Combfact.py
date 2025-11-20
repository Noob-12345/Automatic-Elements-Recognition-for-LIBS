import numpy as np
import os
import glob
import pandas as pd
kB=8.617330350e-5 #eV/K
#计算U（T） 返回U和U总和
def U_Calculate(g,A,E,T):
    U=np.zeros(len(g))
    for i in range(len(g)):
        U[i]=g[i]*np.exp(-E[i]/(kB*T))
    return U,np.sum(U)

#计算相对强度 返回相对强度
#遗留问题1：模拟强度到底要不要wl
def rel_intensity(wl,A,E,g,T):
    U_T,U_T_sum=U_Calculate(g,A,E,T)
    rel_intensity=np.zeros(len(wl))
    for i in range(len(wl)):
        rel_intensity[i]=(A[i]*g[i]*np.exp(-E[i]/(kB*T)))/(U_T_sum*wl[i])  
    return rel_intensity

#元素库制作 返回elements字典和elements_list元素列表
def elements_database(folder_path,T):
    file_list = glob.glob(os.path.join(folder_path, "*.csv"))
    elements_list = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
    #元素特征光谱制作
    elements={}
    for element_name in elements_list: 
        file_path = os.path.join(folder_path, element_name + ".csv")  # 拼接完整路径
        df = pd.read_csv(file_path,header=1,encoding="gbk")  # 读取该元素的csv
        df=df.to_numpy()
        even_rows = df[1::2]
        wl=even_rows[:,1]*0.1
        A=even_rows[:,2]
        E=even_rows[:,3]*1.2398*10**(-4) #eV
        g=even_rows[:,7]
        # 强制转换为 float
        wl = wl.astype(float)
        A  = A.astype(float)
        E  = E.astype(float)
        g  = g.astype(float)
        #波段过滤 200-900nm
        mask = (wl >= 200) & (wl <= 900)
        wl = wl[mask]
        A = A[mask]
        E = E[mask]
        g = g[mask]
        
        relative_intensity=rel_intensity(wl,A,E,g,T)
        matrix = np.column_stack((wl, relative_intensity,A,E,g))
        elements[element_name] = { "data": matrix}
    return elements,elements_list
