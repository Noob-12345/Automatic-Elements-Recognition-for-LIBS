#本文件用于将前面的寻峰代码和模拟谱峰代码结合在一起
#并且自写规则进行置信度计算
#参量解释：elements：元素数据库

import numpy as np
import pandas as pd
import glob
import os
import pywt
import matplotlib.pyplot as plt
from Wavelet_peakfinding import find_peaks1,find_peaks_ridge,peak_correction,wavelet_peak_detection #寻峰

#-----预备-----
#参数设置
T=10000 
kB=8.617330350e-5 #eV/K
#-----数据导入-----
folder_path = r'D:\LIBS\ElementDetectation\LIBS-ElementRecogonise\Code\Elements_database' #元素库路径
data=pd.read_csv(r'D:\LIBS\ElementDetectation\LIBS-ElementRecogonise\Code\SpecSimuDatabase\Cr100_10000K.csv',header=0,skipinitialspace=True)#待测光谱路径
data = data.fillna(0).to_numpy()
data = np.nan_to_num(data, nan=0.0)
x = data[:, 0]
intensity_sum=data[:,1]
signal=data[:,2]
intensity_ionized=data[:,3]



#----必备函数定义----
#计算U（T） 返回U和U总和
def U_Calculate(g,E):
    U=np.zeros(len(g))
    for i in range(len(g)):
        U[i]=g[i]*np.exp(-E[i]/(kB*T))
    return U,np.sum(U)

#计算相对强度 返回相对强度
def rel_intensity(wl,A,E,g):
    U_T,U_T_sum=U_Calculate(g,E)
    rel_intensity=np.zeros(len(wl))
    for i in range(len(wl)):
        rel_intensity[i]=(A[i]*g[i]*np.exp(-E[i]/(kB*T)))/U_T_sum
    return rel_intensity

#元素库制作 返回elements字典和elements_list元素列表
def elements_database(folder_path):
    folder_path = r'D:\LIBS\ElementDetectation\LIBS-ElementRecogonise\Code\Elements_database' #元素库路径
    # 获取所有Excel文件路径
    file_list = glob.glob(os.path.join(folder_path, "*.csv"))
    # 获取元素名字（去掉路径和后缀）
    elements_list = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
    #元素特征光谱制作
    elements={}
    for element_name in elements_list: 
        dfs = []
        file_path = os.path.join(folder_path, element_name + ".csv")  # 拼接完整路径
        df = pd.read_csv(file_path,header=1,encoding="gbk")  # 读取该元素的csv
        df=df.to_numpy()
        even_rows = df[1::2]
        wl=even_rows[:,1]*0.1
        A=even_rows[:,2]
        E=even_rows[:,3]*1.2398*10**(-4) #eV
        g=even_rows[:,7]
    #波段过滤 200-900nm
        mask = (wl >= 200) & (wl <= 900)
        wl = wl[mask]
        A = A[mask]
        E = E[mask]
        g = g[mask]
        
        relative_intensity=rel_intensity(wl,A,E,g)
        matrix = np.column_stack((wl, relative_intensity))
        elements[element_name] = { "data": matrix}
    return elements,elements_list

#计算匹配度（元素库内光谱与待测光谱中元素的匹配度）

#-----主程序-----
elements,elements_list=elements_database(folder_path)
true_peak_idx, peak_wl, peak_int = wavelet_peak_detection(signal,x,wavelet='mexh', scales=np.arange(1, 11), 
                           neighbor=4, min_length=3, coeffi_threshold=1000, window=5)#峰值校正


#-----置信度设置-----
#方案1：波长+强度归一化设置权重
match_results = {}#最终元素置信度存储
for element_name, element_data in elements.items():
    # element_data["data"] 是一个二维矩阵 [波长, 模拟强度]
    element_matrix = element_data["data"]
    element_wl = element_matrix[:, 0]
    element_intensity = element_matrix[:, 1]
    # 这里可以做你想要的对比，比如找落在峰值波长范围内的谱线

    # 波长匹配！！待完善
    wl_min = peak_wl.min()
    wl_max = peak_wl.max()
    mask = (element_wl >= wl_min) & (element_wl <= wl_max)
    matched_wl = element_wl[mask]
    matched_intensity = element_intensity[mask]
    #matched是元素库内落在峰值范围内的谱线
    rel_intensity_sum=np.sum(matched_intensity)

    O_distance=0
    for sim_wl, sim_int in zip(matched_wl, matched_intensity):
        # 找到实际峰值中最接近的波长
        idx = np.argmin(np.abs(peak_wl - sim_wl))
        closest_peak = peak_wl[idx]
        # if element_name=='LiII':
        #     print(sim_wl,closest_peak)
        O_distance+= sim_int*((sim_wl - closest_peak)**2)/rel_intensity_sum # 欧几里得距离
        # if element_name=='CuII':
        #     print(O_distance)

    if O_distance==0:
        O_distance=1e+4 #大值防止出现全部特征谱线都不在峰值范围内的情况

    # 存入字典
    match_results[element_name] = O_distance

# 打印排序后结果
for elem, distance in sorted(match_results.items(), key=lambda x: x[1]):
    print(f"{elem}: 距离(O_distance) = {distance:.4f}")


#debug
target_element = "LiII"   # 想要高亮的元素
target_element2="NaII"

for elem in elements_list:
    if elem == target_element:  
        plt.scatter(elements[elem]['data'][:,0], elements[elem]['data'][:,1], 
                    s=10, color='blue', label=f"{elem} (target)")  
    elif elem == target_element2:  
        plt.scatter(elements[elem]['data'][:,0], elements[elem]['data'][:,1], 
                    s=10, color='orange', label=f"{elem} (target)")
    else:
        plt.scatter(elements[elem]['data'][:,0], elements[elem]['data'][:,1], 
                    s=2, color='green')


plt.plot(x, signal)
plt.scatter(peak_wl, peak_int, color='red')
plt.show()

