#本文件用于将前面的寻峰代码和模拟谱峰代码结合在一起
#并且自写规则进行置信度计算
#参量解释：elements：元素数据库

import numpy as np
import pandas as pd
import glob
import os
import pywt
import matplotlib.pyplot as plt
from collections import defaultdict
from Wavelet_peakfinding import find_peaks1,find_peaks_ridge,peak_correction,wavelet_peak_detection #寻峰

#-----预备-----
#参数设置
T=10000 
kB=8.617330350e-5 #eV/K
#-----数据导入-----
folder_path = r'E:\工作文件\课题组激光诱导击穿光谱学习\LIBS-ElementRecogonise\Latest\Elements_database' #元素库路径
data=pd.read_csv(r'E:\工作文件\课题组激光诱导击穿光谱学习\LIBS-ElementRecogonise\Latest\SpecSimuDatabase\Li100_10000K.csv',header=0,skipinitialspace=True)#待测光谱路径
data = data.fillna(0).to_numpy()
data = np.nan_to_num(data, nan=0.0)
x = data[:, 0]
intensity_sum=data[:,1]
signal=data[:,1]
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
    folder_path = r'E:\工作文件\课题组激光诱导击穿光谱学习\LIBS-ElementRecogonise\Latest\Elements_database' #元素库路径
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


#-----相似度设置-----
#方案1：波长+强度归一化设置权重
def compute_element_confidence(elements, peak_wl):
    match_results = {}#最终元素置信度存储
    element_distance = defaultdict(list)  # 存储每个元素的多个粒子的distance
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
        base_elem = ''.join([c for c in element_name if not c.isdigit() and c not in ["I","V"]])
        element_distance[base_elem].append(O_distance)#元素种类分类的欧几里得距离
        

    # 打印排序后结果
    #粒子
    for elem, distance in sorted(match_results.items(), key=lambda x: x[1]):
        print(f"{elem}: 距离(O_distance) = {distance:.4f}")
    #元素
    for elem, distances in sorted(element_distance.items(), key=lambda x: np.mean(x[1])):
        avg_distance = np.sum(distances)
        print(f"{elem}: 距离(O_distance) = {avg_distance:.4f}")

    return match_results
# a=compute_element_confidence(elements, peak_wl)


#方案二：谱线形状相似度
#遍历每一个元素，在elements_database中提取出谱线的波长并且对应到寻峰结果peak_wl中寻找
#scope为1nm的最近峰，如果超出1nm那很可能是寻峰问题，则舍弃掉该谱线
#将每个元素提出出来的所有谱线与实际峰值进行对比，计算欧几里得距离

def compute_element_confidence_shape(elements, peak_wl, peak_int, scope=0.25):
    """
    方案二：用理论和实验谱形的欧几里得距离作为相似度
    elements: 元素数据库 { "ElemI": {"data": [wl, intensity]} }
    peak_wl: 实验寻峰得到的峰位
    peak_int: 实验寻峰得到的峰强度
    scope: 容许匹配窗口 (nm),默认1nm
    """

    match_results = {}
    element_distance = defaultdict(list)

    #遍历每一个元素
    for element_name, element_data in elements.items():
        element_matrix = element_data["data"]
        element_wl = element_matrix[:, 0]
        element_intensity = element_matrix[:, 1]
        #强度（计算O_distance 用）
        theo_vec = [] 
        exp_vec = []
        # 匹配成功的谱线（波长+强度）(绘图用)
        matched_theo = []  # 保存匹配成功的理论谱线
        matched_exp = []   # 保存匹配成功的实验谱线

        # 初始化实验峰匹配标记
        matched_flag = np.zeros(len(peak_wl), dtype=bool)

        for sim_wl, sim_int in zip(element_wl, element_intensity):
            # 找到最接近的实验峰
            available_idx = np.where(~matched_flag)[0]
            if len(available_idx) == 0:
                theo_vec.append(sim_int)
                exp_vec.append(0)
                continue


            nearest_idx = available_idx[np.argmin(np.abs(peak_wl[available_idx] - sim_wl))]
            diff = abs(peak_wl[nearest_idx] - sim_wl)

            if diff <= scope:
                # 匹配成功
                theo_vec.append(sim_int)
                exp_vec.append(peak_int[nearest_idx])
                matched_theo.append((sim_wl, sim_int))
                matched_exp.append((peak_wl[nearest_idx], peak_int[nearest_idx]))
                matched_flag[nearest_idx] = True
            else:
                # 匹配失败：理论有谱线，实验没有 → 实验强度记为0 （匹配失败策略待完善）
                theo_vec.append(sim_int)#（可以设置为0或者是平均值什么的）
                exp_vec.append(0)

        if len(matched_exp) == 0: #（极端情况，无匹配峰）
            O_distance = 1e4
        else:
            theo_vec = np.array(theo_vec)
            exp_vec = np.array(exp_vec)
            N_total = len(element_wl)
            N_matched = len(matched_exp)
            match_ratio = N_matched / N_total if N_total > 0 else 0 # 匹配率
            # print(f"{element_name}: 匹配率 = {match_ratio:.2f}, 匹配峰数 = {N_matched}, 总谱线数 = {N_total}")

            # 归一化
            if np.sum(theo_vec) > 0:
                theo_vec = theo_vec / np.sum(theo_vec)
            if np.sum(exp_vec) > 0:
                exp_vec = exp_vec / np.sum(exp_vec)
            O_distance =(np.sqrt(np.sum((theo_vec - exp_vec) ** 2)))/(0.03 + match_ratio)  # 考虑匹配率的影响 0.03防止除0

        match_results[element_name] = O_distance
        base_elem = ''.join([c for c in element_name if not c.isdigit() and c not in ["I","V"]])
        element_distance[base_elem].append(O_distance)
        if element_name == 'LiI':
            plt.figure(figsize=(8,4))

        # 全部理论谱线（浅蓝）
            all_theo_intensity = element_intensity / np.sum(element_intensity)
            for wl, inten_norm in zip(element_wl, all_theo_intensity):
                plt.vlines(wl, 0, inten_norm,
                        color='lightblue', alpha=0.5,
                        label='All Theoretical' if wl==element_wl[0] else "")

            # 理论匹配谱线（蓝）
            if matched_theo:
                matched_theo_intensity = np.array([inten for _, inten in matched_theo])
                matched_theo_norm = matched_theo_intensity / np.sum(matched_theo_intensity)
                for (wl, _), inten_norm in zip(matched_theo, matched_theo_norm):
                    plt.vlines(wl, 0, inten_norm,
                            color='b', alpha=0.7,
                            label='Matched Theoretical' if wl==matched_theo[0][0] else "")

            # --- 匹配成功的实验谱线（红色） ---
            if matched_exp:
                matched_exp_intensity = np.array([inten for _, inten in matched_exp])
                matched_exp_norm = matched_exp_intensity / np.sum(matched_exp_intensity)
                for (wl, _), inten_norm in zip(matched_exp, matched_exp_norm):
                    plt.vlines(wl, 0, inten_norm,
                            color='r', alpha=0.7,
                            label='Matched Experimental' if wl==matched_exp[0][0] else "")

            plt.title(f'Matched Stick Spectrum for {element_name}')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Normalized Intensity')
            plt.legend()
            plt.show()


    #粒子
    print("\n--- 粒子层面 ---")
    for elem, distance in sorted(match_results.items(), key=lambda x: x[1]):
        print(f"{elem}: 距离 = {distance:.4f}")

    # 元素
    print("\n--- 元素层面 ---")
    for elem, distances in sorted(element_distance.items(), key=lambda x: np.mean(x[1])):
        avg_distance = np.mean(distances)
        print(f"{elem}: 平均距离 = {avg_distance:.4f}")

    return match_results
b=compute_element_confidence_shape(elements, peak_wl, peak_int, scope=0.25)




# #debug
# target_element = "CrII"   # 想要高亮的元素
# target_element2="NaII"

# for elem in elements_list:
#     if elem == target_element:  
#         plt.scatter(elements[elem]['data'][:,0], elements[elem]['data'][:,1], 
#                     s=10, color='blue', label=f"{elem} (target)")  
#     elif elem == target_element2:  
#         plt.scatter(elements[elem]['data'][:,0], elements[elem]['data'][:,1], 
#                     s=10, color='orange', label=f"{elem} (target)")
#     else:
#         plt.scatter(elements[elem]['data'][:,0], elements[elem]['data'][:,1], 
#                     s=2, color='green')



#-----置信度设置-----
#方案一：整体相似度反比归一化   (1/distance)/sum(1/distance)
# distance_sum=0
# for distance in sorted(a.values()):
#     distance_sum+=1/distance

# for elem, distance in sorted(a.items(), key=lambda x: x[1]):
#     confidence=(1/distance)/distance_sum
#     print(f"{elem}: 置信度 = {confidence:.4f}")


# plt.plot(x, signal)
# plt.scatter(peak_wl, peak_int, color='red')
# plt.show()

