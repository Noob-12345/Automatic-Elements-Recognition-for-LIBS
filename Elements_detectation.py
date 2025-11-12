
import numpy as np
import pandas as pd
import glob
import os
import pywt
import matplotlib.pyplot as plt
from collections import defaultdict
from Wavelet_peakfinding import find_peaks_ridge,peak_correction,wavelet_peak_detection #寻峰

#-----预备-----
#参数设置
T=10000 
kB=8.617330350e-5 #eV/K
#-----数据导入-----
folder_path = r'D:\LIBS\ElementDetectation\11.10\Elements_database' #元素库路径

#----必备函数定义----
#计算U（T） 返回U和U总和
def U_Calculate(g,A,E):
    U=np.zeros(len(g))
    for i in range(len(g)):
        U[i]=g[i]*np.exp(-E[i]/(kB*T))
    return U,np.sum(U)

#计算相对强度 返回相对强度
#遗留问题1：模拟强度到底要不要wl
def rel_intensity(wl,A,E,g):
    U_T,U_T_sum=U_Calculate(g,A,E)
    rel_intensity=np.zeros(len(wl))
    for i in range(len(wl)):
        rel_intensity[i]=(A[i]*g[i]*np.exp(-E[i]/(kB*T)))/(U_T_sum*wl[i])  
    return rel_intensity

#元素库制作 返回elements字典和elements_list元素列表
def elements_database(folder_path):
    folder_path = r'D:\LIBS\ElementDetectation\11.10\Elements_database' #元素库路径
    # 获取所有Excel文件路径
    file_list = glob.glob(os.path.join(folder_path, "*.csv"))
    # 获取元素名字（去掉路径和后缀）
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
        
        relative_intensity=rel_intensity(wl,A,E,g)
        matrix = np.column_stack((wl, relative_intensity,A,E,g))
        elements[element_name] = { "data": matrix}
    return elements,elements_list

#玻尔兹曼图拟合 返回斜率，截距，温度，y
def Boltzmann_fit(I, wl,A, g, E):
    y = np.log(I*wl/ (g * A))

    # 线性拟合
    coefficients = np.polyfit(E, y, 1) #slope斜率 intercpet截距 拟合
    slope, intercept = coefficients
    T = -1/(slope * kB)  # 温度计算
    return slope, intercept, T, y

def Boltzmann_plot(matched_i, matched_wl, element_A, element_E, element_g, element_wl,element_name):

#参数说明:matched_theo匹配到的理论谱线  matched_exp匹配到的实验谱线  element_A元素的A  element_E元素的E  element_g元素的g  element_wl元素的波长列表  element_name元素名称
#用途说明：检测匹配点并且绘制玻尔兹曼图
    # ====== ② 玻尔兹曼图计算与绘制 ======
    if len(matched_wl) > 2:  # 至少3个点才能线性拟合
        print(f"\n--- {element_name} 玻尔兹曼图 ---")
        
        # 提取匹配到的谱线参数（与 matched_exp 对应的理论参数）
        matched_wl = np.array([t[0] for t in matched_wl])
        matched_I = np.array([t[1] for t in matched_i])  # 实验强度
        # 从理论库中取对应的 A、E、g
        matched_idx = [np.argmin(np.abs(element_wl - wl)) for wl in matched_wl]
        A_sel = element_A[matched_idx]
        E_sel = element_E[matched_idx]
        g_sel = element_g[matched_idx]


        matched_I = np.array(matched_I, dtype=float)
        A_sel = np.array(A_sel, dtype=float)
        g_sel = np.array(g_sel, dtype=float)
        E_sel = np.array(E_sel, dtype=float)

        # 玻尔兹曼拟合
        slope, intercept, T_fit, y = Boltzmann_fit(matched_I, matched_wl,A_sel, g_sel, E_sel)
        print(f"拟合温度 T = {T_fit:.2f} K, 斜率 = {slope:.3f}")
        
        # 绘图
        # plt.figure(figsize=(6,4))
        # plt.scatter(E_sel, y, c='r', label='Points')
        # plt.plot(E_sel, slope*E_sel + intercept, 'b--', label=f'fit: T={T_fit:.1f} K')
        # plt.xlabel('E (eV)')
        # plt.ylabel('ln(I / (g·A))')
        # plt.title(f'{element_name} Boltzmann Plot')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print(f"{element_name} 匹配峰数不足，无法绘制玻尔兹曼图。")

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
    final_results = {} #元素层面显示
    #遍历每一个粒子
    for element_name, element_data in elements.items():
        element_matrix = element_data["data"]
        element_wl = element_matrix[:, 0]
        element_intensity = element_matrix[:, 1]
        element_A=element_matrix[:,2]
        element_E=element_matrix[:,3]
        element_g=element_matrix[:,4]
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
                theo_vec.append(0)#（可以设置为0或者是平均值什么的）
                exp_vec.append(0)


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
            if O_distance ==0: #完全没谱线或者只有一条谱线的时候
                O_distance=1e+4

        match_results[element_name] = O_distance
        base_elem = ''.join([c for c in element_name if not c.isdigit() and c not in ["I","V"]])
        element_distance[base_elem].append(O_distance)

        if element_name == 'CaII':
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
                for (wl, _), inten_norm_theo in zip(matched_theo, matched_theo_norm):
                    plt.vlines(wl, 0, inten_norm_theo,
                            color='b', alpha=0.7,
                            label='Matched Theoretical' if wl==matched_theo[0][0] else "")

            # --- 匹配成功的实验谱线（红色） ---
            if matched_exp:
                matched_exp_intensity = np.array([inten for _, inten in matched_exp])
                matched_exp_norm = matched_exp_intensity / np.sum(matched_exp_intensity)
                matched_exp_normalized = [(wl, inten_norm_exp)
                          for (wl, _), inten_norm_exp in zip(matched_exp, matched_exp_norm)]
                for (wl, _), inten_norm_exp in zip(matched_exp, matched_exp_norm):
                    plt.vlines(wl, 0, inten_norm_exp,
                            color='r', alpha=0.7,
                            label='Matched Experimental' if wl==matched_exp[0][0] else "")



            plt.title(f'Matched Stick Spectrum for {element_name}')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Normalized Intensity')
            Boltzmann_plot(matched_exp, matched_theo, element_A, element_E, element_g, element_wl,element_name)
            plt.show()



#---------------Debug分割线-------------------------
#元素置信度判断
    for base_elem, distances in element_distance.items():
        min_distance = min(distances)
        if min_distance < 0.2:  # 阈值可调
            final_results[base_elem] = min_distance
        else:
            final_results[base_elem] = np.mean(distances)
    
    return match_results,final_results


#-----主程序-----
elements,elements_list=elements_database(folder_path)

signal_path= r'D:\LIBS\ElementDetectation\11.10\SpecSimuDatabase' 
I_file_list = glob.glob(os.path.join(signal_path, "*.csv"))
I_elements_list = [os.path.splitext(os.path.basename(f))[0] for f in I_file_list]
target_files=['Mg100_10000K_PF']
for I_element_name in I_elements_list:

    if I_element_name not in target_files:
        continue  # 跳过不在名单内的文件
    data=pd.read_csv(os.path.join(signal_path, I_element_name + ".csv"),header=0,skipinitialspace=True)#待测光谱路径
    data = data.fillna(0).to_numpy()
    data = np.nan_to_num(data, nan=0.0)
    x = data[:, 0]
    intensity_sum=data[:,1]
    signal=data[:,1]
    intensity_ionized=data[:,3]
    true_peak_idx, peak_wl, peak_int = wavelet_peak_detection(signal,x,wavelet='mexh', scales=np.arange(1, 11), 
                               neighbor=4, min_length=3, coeffi_threshold=1000, window=5)#峰值校正

    particle_result,elements_result=compute_element_confidence_shape(elements, peak_wl, peak_int, scope=0.25)
    print("\n---" ,I_element_name, "---") 
    # 粒子
    print("--- 粒子层面 ---\n")
    for elem, distance in sorted(particle_result.items(), key=lambda x: x[1]):
        print(f"{elem}: 距离 = {distance:.4f}")

    # 元素
    print("--- 元素层面 ---")
    for elem, distances in sorted(elements_result.items(), key=lambda x: np.mean(x[1])):
        print(f"{elem}: 平均距离 = {distances:.4f}")





