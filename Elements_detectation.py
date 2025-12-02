
import numpy as np
import pandas as pd
import glob
import os
import pywt
import matplotlib.pyplot as plt
from collections import defaultdict
from Wavelet_peakfinding import find_peaks_ridge,peak_correction,wavelet_peak_detection #寻峰
from Elements_Combfact import elements_database #元素库制作
from scipy.optimize import linear_sum_assignment
#-----预备-----
#参数设置
T=10000 
kB=8.617330350e-5 #eV/K
#-----数据导入-----
folder_path = r'D:\LIBS\ElementDetectation\11.10\Elements_database' #元素库路径

#----必备函数定义----
#玻尔兹曼图拟合 返回斜率，截距，温度，y
def Boltzmann_fit(I, wl, A, g, E):
    y = np.log(I*wl / (g * A))
    
    # 线性拟合
    coefficients = np.polyfit(E, y, 1)  # slope斜率 intercept截距
    slope, intercept = coefficients
    T = -1 / (slope * kB)  # 温度计算
    
    # 计算 R²
    y_fit = slope * E + intercept
    ss_res = np.sum((y - y_fit) ** 2)  # 残差平方和
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # 总平方和
    R2 = 1 - (ss_res / ss_tot)
    
    return slope, intercept, T, R2, y

def Boltzmann_fit_iterative(I, wl, A, g, E,R2_threshold=1e-1,R2_start_threshold=0.97,max_iter=5,verbose=False):
    """
    迭代 Boltzmann 拟合（删除偏差最大点）
    同步删除所有变量：I, wl, A, g, E
    """

    # 转 numpy
    I = np.array(I, float)
    wl = np.array(wl, float)
    A = np.array(A, float)
    g = np.array(g, float)
    E = np.array(E, float)

    N = len(E)
    idx = np.arange(N)

    # ---- 初次拟合 ----
    y = np.log(I * wl / (g * A))

    slope, intercept = np.polyfit(E, y, 1)
    y_pred = slope * E + intercept

    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    R2_init = 1 - ss_res / ss_tot

    if verbose:
        print(f"[Init] R²={R2_init:.5f}")

    # ---- 若初始 R² 已够好 → 不迭代 ----
    if R2_init >= R2_start_threshold:
        T = -1/(slope*kB)
        return slope, intercept, T, R2_init, \
               y, E, wl, I, A, g

    # ---- 迭代删除最差点 ----
    R2_prev = R2_init

    for it in range(max_iter):

        y = np.log(I * wl / (g * A))
        y_pred = slope * E + intercept
        residuals = y - y_pred

        # 找到偏差最大的点
        worst = np.argmax(np.abs(residuals))

        if verbose:
            print(f"[Iter {it+1}] remove index {worst}, resid={residuals[worst]:.5f}")

        # 同步删除所有变量
        I = np.delete(I, worst)
        wl = np.delete(wl, worst)
        A = np.delete(A, worst)
        g = np.delete(g, worst)
        E = np.delete(E, worst)

        if len(E) < 2:
            break

        # 重新拟合
        y = np.log(I * wl / (g * A))
        slope, intercept = np.polyfit(E, y, 1)
        y_pred = slope * E + intercept

        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        R2_new = 1 - ss_res / ss_tot

        delta_R2 = abs(R2_new - R2_prev)

        if verbose:
            print(f"    R²={R2_new:.5f}, ΔR²={delta_R2:.6f}")

        if delta_R2 < R2_threshold:
            break

        R2_prev = R2_new

    # ---- 最终温度 ----
    T = -1/(slope*kB)

    # 返回所有删点后的数据
    return slope, intercept, T, R2_prev, \
           np.log(I * wl / (g * A)), E, wl, I, A, g

def Boltzmann_plot(matched_i, matched_wl, element_A, element_E, element_g, element_wl,element_name):

#参数说明:matched_theo匹配到的理论谱线  matched_exp匹配到的实验谱线  element_A元素的A  element_E元素的E  element_g元素的g  element_wl元素的波长列表  element_name元素名称
#用途说明：检测匹配点并且绘制玻尔兹曼图
    # ====== ② 玻尔兹曼图计算与绘制 ======
    if len(matched_wl) >= 2:  # 至少3个点才能线性拟合
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
        slope, intercept, T_fit,R2, y_full = Boltzmann_fit(matched_I, matched_wl,A_sel, g_sel, E_sel)
        # slope, intercept, T_fit,R2, y_full, y_used = Boltzmann_fit_iterative(matched_I, matched_wl,A_sel, g_sel, E_sel,R2_start_threshold=0.97,max_iter=5,verbose=False)
        # print(f"拟合温度 T = {T_fit:.2f} K, 斜率 = {slope:.3f}")
        
        #绘图
        slope, intercept, T_fit, R2, y_used, E_used, wl_used, I_used, A_used, g_used = \
            Boltzmann_fit_iterative(matched_I, matched_wl, A_sel, g_sel, E_sel,
                                    R2_start_threshold=0.1, max_iter=1, verbose=False)

        plt.figure(figsize=(6,4))
        plt.scatter(E_used, y_used, c='r', label='Used Points')
        plt.plot(E_used, slope * E_used + intercept, 'b--',label=f'Fit T={T_fit:.1f} K, R2={R2:.3f}')
        plt.xlabel('E (eV)')
        plt.ylabel('ln(I / (g·A))')
        plt.title(f'{element_name} Boltzmann Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    else:
        print(f"{element_name} 匹配峰数不足，无法绘制玻尔兹曼图。")

#匹配峰策略
def used_match_spectral_lines(scope):
        #遍历每一个粒子
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
                theo_vec.append(0)
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

        theo_vec = np.array(theo_vec)
        exp_vec = np.array(exp_vec)
        N_total = len(element_wl)
        N_matched = len(matched_exp)
        match_ratio = N_matched / N_total if N_total > 0 else 0 # 匹配率

    return theo_vec, exp_vec, matched_theo, matched_exp
        
#匈牙利算法线匹配策略
def match_spectral_lines(theo_wl, theo_int, exp_wl, exp_int, scope):

    T = len(theo_wl)
    E = len(exp_wl)
    N = max(T, E) 
    cost = np.zeros((N, N), dtype=float)
    BIG = 1e6
    cost[:] = BIG
    for i in range(T):
        for j in range(E):
            cost[i, j] = abs(theo_wl[i] - exp_wl[j])

    #  匈牙利算法
    row_ind, col_ind = linear_sum_assignment(cost)

    theo_vec = []
    exp_vec = []

    matched_theo = []
    matched_exp = []

    for i, j in zip(row_ind, col_ind):

        if i < T:            # 这是一个真实的理论峰
            if j < E:        # 实验峰也是真实的
                diff = abs(theo_wl[i] - exp_wl[j])

                if diff <= scope:
                    # 匹配成功
                    theo_vec.append(theo_int[i])
                    exp_vec.append(exp_int[j])

                    matched_theo.append((theo_wl[i], theo_int[i]))
                    matched_exp.append((exp_wl[j], exp_int[j]))
                else:
                    # 匹配距离太大 → 当作未匹配
                    theo_vec.append(0)
                    exp_vec.append(0)

            else:
                # 实验峰不存在（补的虚拟列）→ 未匹配
                theo_vec.append(0)
                exp_vec.append(0)

    theo_vec = np.array(theo_vec)
    exp_vec = np.array(exp_vec)

    return theo_vec, exp_vec, matched_theo, matched_exp

def match_spectral_lines_weighted(theo_wl, theo_int, exp_wl, exp_int, scope=0.2, max_high_intensity=None, alpha=1.0, beta=1.0):
    """
    匈牙利算法改进：优先匹配高强度实验峰
    
    Parameters:
        theo_wl, theo_int : 理论谱线波长和强度
        exp_wl, exp_int   : 实验谱线波长和强度
        scope             : 匹配容差 (nm)
        max_high_intensity: 限制参与匹配的高强度峰数量
        alpha, beta       : 成本权重，cost = alpha*|wl_diff| - beta*exp_intensity
    """
    exp_wl = np.array(exp_wl, dtype=float)
    exp_int = np.array(exp_int, dtype=float)
    theo_wl = np.array(theo_wl, dtype=float)
    theo_int = np.array(theo_int, dtype=float)

    # 按实验强度排序，选出前 N 个强峰
    exp_idx_use = np.arange(len(exp_wl))
    if max_high_intensity is not None and len(exp_wl) > max_high_intensity:
        exp_idx_use = np.argsort(-exp_int)[:max_high_intensity]
    
    exp_wl_sel = exp_wl[exp_idx_use]
    exp_int_sel = exp_int[exp_idx_use]
    
    T = len(theo_wl)
    E = len(exp_wl_sel)
    N = max(T, E)
    BIG = 1e6
    
    # 构建成本矩阵
    cost = np.full((N, N), BIG, dtype=float)
    for i in range(T):
        for j in range(E):
            diff = abs(theo_wl[i] - exp_wl_sel[j])
            if diff <= scope:
                cost[i, j] = alpha * diff - beta * exp_int_sel[j]  # 波长差减去强度加权
    
    # 匈牙利算法
    row_ind, col_ind = linear_sum_assignment(cost)
    
    matched_theo = []
    matched_exp = []
    theo_vec = []
    exp_vec = []
    
    for i, j in zip(row_ind, col_ind):
        if i < T and j < E and cost[i,j] < BIG:
            # 匹配成功
            matched_theo.append((theo_wl[i], theo_int[i]))
            matched_exp.append((exp_wl_sel[j], exp_int_sel[j]))
            theo_vec.append(theo_int[i])
            exp_vec.append(exp_int_sel[j])
        else:
            # 未匹配
            theo_vec.append(0)
            exp_vec.append(0)
    
    return np.array(theo_vec), np.array(exp_vec), matched_theo, matched_exp


#方案二：谱线形状相似度
#遍历每一个元素，在elements_database中提取出谱线的波长并且对应到寻峰结果peak_wl中寻找
#scope为1nm的最近峰，如果超出1nm那很可能是寻峰问题，则舍弃掉该谱线
#将每个元素提出出来的所有谱线与实际峰值进行对比，计算欧几里得距离
def compute_element_confidence_shape(elements, peak_wl, peak_int,global_wl,global_intensity,scope=0.2):
    """
    方案二：用理论和实验谱形的欧几里得距离作为相似度
    elements: 元素数据库 { "ElemI": {"data": [wl, intensity]} }
    peak_wl: 实验寻峰得到的峰位
    peak_int: 实验寻峰得到的峰强度
    scope: 容许匹配窗口 (nm),默认1nm
    """

    match_results = {}
    element_distance = defaultdict(list)
    element_T = defaultdict(list)
    element_R2 = defaultdict(list)
    final_results = {} #元素层面显示
    final_T={}
    final_R2={}
    Boltzmann_T={}
    Boltzmann_R2={}


    #遍历每一个粒子
    for element_name, element_data in elements.items():
        element_matrix = element_data["data"]
        element_wl = element_matrix[:, 0]
        element_intensity = element_matrix[:, 1]
        element_A=element_matrix[:,2]
        element_E=element_matrix[:,3]
        element_g=element_matrix[:,4]

        theo_vec, exp_vec, matched_theo, matched_exp = match_spectral_lines_weighted(element_wl, element_intensity, peak_wl, peak_int, scope)

        theo_vec = np.array(theo_vec)
        exp_vec = np.array(exp_vec)
        N_total = len(element_wl)
        N_matched = len(matched_exp)
        match_ratio = N_matched / N_total if N_total > 0 else 0 # 匹配率
        

        # 归一化
        if np.sum(theo_vec) > 0:
            theo_vec = theo_vec / np.sum(theo_vec)
        if np.sum(exp_vec) > 0:
            exp_vec = exp_vec / np.sum(exp_vec)

        O_distance =(np.sqrt(np.sum((theo_vec - exp_vec) ** 2)))/(0.03 + match_ratio)  # 考虑匹配率的影响 0.03防止除0   
        if O_distance ==0: #完全没谱线或者只有一条谱线的时候
            O_distance=1e+4

        #BoltzmannT，R2计算
        if len(matched_theo) >= 2:
            # 提取匹配到的谱线参数（与 matched_exp 对应的理论参数）
            matched_wl = np.array([t[0] for t in matched_theo])
            matched_I = np.array([t[1] for t in matched_exp])  # 实验强度
            # 从理论库中取对应的 A、E、g
            matched_idx = [np.argmin(np.abs(element_wl - wl)) for wl in matched_wl]
            slope, intercept, T_fit, R2, y  = Boltzmann_fit(matched_I, matched_wl, element_A[matched_idx], element_g[matched_idx], element_E[matched_idx])
            slope,intecept,T_fit_iterative,R2_itertative,y,E_iterative,wl_iterative,I_iterative,A_iterative,g_iterative=Boltzmann_fit_iterative(matched_I, matched_wl, element_A[matched_idx], element_g[matched_idx], element_E[matched_idx],R2_start_threshold=0.97, max_iter=3, verbose=False)
            
            Boltzmann_T[element_name] = T_fit
            Boltzmann_R2[element_name] = R2
        else:
            Boltzmann_T[element_name] = 0
            Boltzmann_R2[element_name] = 0

        if element_name == 'CuI':
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
                    
            ### --- 新增波形标注逻辑 --- ###
            plt.figure(figsize=(10,4))
            plt.plot(global_wl, global_intensity, color='black', lw=1, label='Original Spectrum')

            # 标出所有理论谱线位置（浅蓝色线）
            for wl in element_wl:
                plt.axvline(wl, color='cyan', alpha=0.3)

            # 标出匹配到的实验峰（红色点）
            for wl, inten in matched_exp:
                plt.scatter(wl, inten, color='red', s=25)

            plt.title('Original Spectrum with CrII Peaks Marked')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Intensity')

            plt.legend(loc='upper right')


            plt.title(f'Matched Stick Spectrum for {element_name}')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Normalized Intensity')
            Boltzmann_plot(matched_exp, matched_theo, element_A, element_E, element_g, element_wl,element_name)
            iterative_combined = np.column_stack((wl_iterative, I_iterative))
            Boltzmann_plot(iterative_combined, iterative_combined, A_iterative, E_iterative, g_iterative, wl_iterative,element_name+"_iterative")
            plt.show()

        match_results[element_name] = O_distance
        base_elem = ''.join([c for c in element_name if not c.isdigit() and c not in ["I","V"]])
        element_T[base_elem].append(Boltzmann_T[element_name])
        element_R2[base_elem].append(Boltzmann_R2[element_name])
        element_distance[base_elem].append(O_distance)


#筛选
#元素距离筛选
    for base_elem, distances in element_distance.items():
        min_distance = min(distances)
        if min_distance < 47.13333:  
            final_results[base_elem] = min_distance
        else:
            final_results[base_elem] = np.mean(distances)

#元素T和R2输出
    for base_elem, Ts in element_T.items():
        # 过滤出大于0的温度
        valid_T = [t for t in Ts if t > 0]

        if valid_T:
            # 找出所有符合条件的 (T, R2) 对
            TR_pairs = []
            if base_elem in element_R2:
                R2s = element_R2[base_elem]
                # 遍历 Ts 列表，筛选出有效温度对应的 R²
                for t, r2 in zip(Ts, R2s):
                    if t > 0 and r2!=1:  # 初筛T>0,R2!=1
                        TR_pairs.append((t, r2))
            
            if TR_pairs:
                # 选出 R² 最大的那一组
                selected_T, selected_R2 = max(TR_pairs, key=lambda x: x[1])
            else:
                # 如果没有 R² 对应信息，就退化为取最小温度
                selected_T = min(valid_T)
                selected_R2 = 0

            # 保存结果
            final_T[base_elem] = selected_T
            final_R2[base_elem] = selected_R2

        else:
            final_T[base_elem] = 0
            final_R2[base_elem] = 0


#反归一化置信度输出
    elements_confidence={}
    for elem, distances in final_results.items():
        if distances<10000:
            #elements_confidence[elem]=1/(1+distances) #倒数映射
            elements_confidence[elem]=np.exp(-1.5*distances/final_R2[elem]) #指数映射
            if final_T[elem]<5000 or final_T[elem]>20000: #电子温度判据
                elements_confidence[elem]=0
        else:
            elements_confidence[elem]=0

    return match_results,final_results,final_T,final_R2,elements_confidence


#-----主程序-----
elements,elements_list=elements_database(folder_path,T)
signal_path= r'D:\LIBS\ElementDetectation\11.10\SpecSimuDatabase' #待测光谱路径
I_file_list = glob.glob(os.path.join(signal_path, "*.csv"))
I_elements_list = [os.path.splitext(os.path.basename(f))[0] for f in I_file_list]
target_files=['904L_10000K_PF']
for I_element_name in I_elements_list:

    if I_element_name not in target_files:
        continue 
    data=pd.read_csv(os.path.join(signal_path, I_element_name + ".csv"),header=0,skipinitialspace=True)#待测光谱路径
    data = data.fillna(0).to_numpy()
    data = np.nan_to_num(data, nan=0.0)
    x = data[:, 0]
    intensity_sum=data[:,1]
    signal=data[:,1]
    intensity_ionized=data[:,3]
    true_peak_idx, peak_wl, peak_int = wavelet_peak_detection(signal,x,wavelet='mexh', scales=np.arange(1, 11), 
                               neighbor=4, min_length=3, coeffi_threshold=1000, window=5)#峰值校正

    particle_result,elements_result,elements_T,elements_R2,elements_confidence=compute_element_confidence_shape(elements, peak_wl, peak_int,x,intensity_sum,scope=0.15)
    print("\n---" ,I_element_name, "---") 
    # # # 粒子
    # print("--- 粒子层面 ---\n")
    # for elem, distance in sorted(particle_result.items(), key=lambda x: x[1]):
    #     print(f"{elem}: 距离 = {distance:.4f}")

    # 元素+置信度
    print("--- 元素层面（距离 + 置信度） ---")
    for elem in sorted(elements_result.keys(), key=lambda x: elements_result[x]):
        dist = elements_result.get(elem, np.nan)
        conf = elements_confidence.get(elem, 0)
        T= elements_T.get(elem, 0)
        R2= elements_R2.get(elem, 0)
        print(f"{elem:<6s} 平均距离 = {dist:<8.4f} | 温度={T:<8.4f}| R2 = {R2:<8.4f}| 置信度 = {conf:<8.4f}")






