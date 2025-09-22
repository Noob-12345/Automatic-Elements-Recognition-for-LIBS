#本文件用于将前面的寻峰代码和模拟谱峰代码结合在一起
#并且自写规则进行置信度计算

import numpy as np
import pandas as pd
import glob
import os
import pywt
import matplotlib.pyplot as plt
from Wavelet_peakfinding import find_peaks1,find_peaks_ridge #寻峰
#-----预备-----
#参数设置
T=10000 
kB=8.617330350e-5 #eV/K
#计算U（T）
def U_Calculate(g,E):
    U=np.zeros(len(g))
    for i in range(len(g)):
        U[i]=g[i]*np.exp(-E[i]/(kB*T))
    return U,np.sum(U)
#计算相对强度
def rel_intensity(wl,A,E,g):
    U_T,U_T_sum=U_Calculate(g,E)
    rel_intensity=np.zeros(len(wl))
    for i in range(len(wl)):
        rel_intensity[i]=(A[i]*g[i]*np.exp(-E[i]/(kB*T)))/U_T_sum
    return rel_intensity
folder_path = r'E:\工作文件\课题组激光诱导击穿光谱学习\LIBS-ElementRecogonise\Code\Elements_database' #数据库路径
# 获取所有Excel文件路径
file_list = glob.glob(os.path.join(folder_path, "*.xlsx"))
# 获取元素名字（去掉路径和后缀）
elements_list = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
#基底制作
elements={}
for element_name in elements_list: 
    dfs = []
    file_path = os.path.join(folder_path, element_name + ".xlsx")  # 拼接完整路径
    df = pd.read_excel(file_path,header=1)  # 读取该元素的Excel
    df=df.to_numpy()
    even_rows = df[1::2]
    wl=even_rows[:,1]*0.1
    A=even_rows[:,2]
    E=even_rows[:,3]*1.2398*10**(-4) #eV
    g=even_rows[:,7]
    relative_intensity=rel_intensity(wl,A,E,g)
    matrix = np.column_stack((wl, relative_intensity))
    elements[element_name] = { "data": matrix}


#-----数据导入-----
test_data=pd.read_excel('test_data.xlsx',header=1)
test_data=test_data.to_numpy()
x=test_data[8500:10591,0]
signal=test_data[8500:10591,5]

#-----小波变换-----
wavelet = 'mexh'  # 小波函数
scales = np.arange(1, 30)  # 尺度范围 (1=窄峰, 大=宽峰)
coefficients, frequencies = pywt.cwt(signal, scales, wavelet)

#-----寻峰-----
ridges_found=find_peaks_ridge(signal,coefficients,neighbor=3,min_length=3,coeffi_threshold=100) #小波脊线寻峰
#峰线查重
peak_ridgefound_index=[]
for ridge in ridges_found:
    min_scale_pos=min(ridge,key=lambda x:x[0])
    scale_idx,pos_idx=min_scale_pos
    if np.isfinite(pos_idx) and int(pos_idx) not in peak_ridgefound_index:  # 去重
        peak_ridgefound_index.append(int(pos_idx))
#提取峰值
peak_wl = x[peak_ridgefound_index]
peak_intensity = signal[peak_ridgefound_index]
peak_found = np.column_stack((peak_wl, peak_intensity))

#-----置信度设置-----
#现在有谱线峰值peak_found和元素库elements

#方案1：特征向量[波长，强度]

match_results = {}#最终元素置信度存储

for element_name, element_data in elements.items():
    # element_data["data"] 是一个二维矩阵 [波长, 模拟强度]
    element_matrix = element_data["data"]
    element_wl = element_matrix[:, 0]
    element_intensity = element_matrix[:, 1]
    # 这里可以做你想要的对比，比如找落在峰值波长范围内的谱线
    # 波长匹配
    wl_min = peak_wl.min()
    wl_max = peak_wl.max()
    mask = (element_wl >= wl_min) & (element_wl <= wl_max)
    matched_wl = element_wl[mask]
    matched_intensity = element_intensity[mask]
    
    # print(matched_wl,matched_intensity)

    O_distance=0

    for sim_wl, sim_int in zip(matched_wl, matched_intensity):
        # 找到实际峰值中最接近的波长
        idx = np.argmin(np.abs(peak_wl - sim_wl))
        closest_peak = peak_wl[idx]
        # print(sim_wl,closest_peak)
        O_distance+= (sim_wl - closest_peak)**2+(sim_int-peak_intensity[idx])**2 # 欧几里得距离

    # 存入字典
    match_results[element_name] = O_distance

# 打印结果
for elem, distance in match_results.items():
    print(f"{elem}: 置信度(O_distance) = {distance:.4f}")
    

       











# print(peak_found)
# print(elements['CrI']['data'])