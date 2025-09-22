#小波变换寻峰算法
#效果：输入光谱数据，输出峰值位置和大小，无论展宽。
#寻峰方式：脊线寻峰

#待解决问题：1、参数设置（尺度，领域半径，最小脊线长度） 2、脊线校正未作 


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt

# ====== 1.模拟信号 ======
# x = np.linspace(0, 100, 500)
# signal = (np.exp(-(x-30)**2/20) +
#           0.8*np.exp(-(x-70)**2/10))  # 高斯峰 + 噪声
test_data=pd.read_excel('test_data.xlsx',header=1)
test_data=test_data.to_numpy()
x=test_data[8500:10591,0] #300-340nm
signal=test_data[8500:10591,5]

# ====== 2. 定义小波参数 ======
wavelet = 'mexh'  # 小波函数
scales = np.arange(1, 30)  # 尺度范围 (1=窄峰, 大=宽峰)
coefficients, frequencies = pywt.cwt(signal, scales, wavelet)
# coefficients.shape = (len(scales), len(signal))

#寻峰策略1：阈值寻峰(尺度<阈值)
def find_peaks1(signal,coefficients,threshold):
    peaks = []
    for i in range(len(signal)):

        coefficients_i=coefficients[:,i]
        if np.max(coefficients_i)>threshold:
            peaks.append(i)
    return peaks


#寻峰策略2：脊线寻峰
#策略：从最大尺度的第一个极大值点开始描点
#signal为原始信号，coefficients为小波系数，neighbor为邻域半径，min_length为最小脊线长度
def find_peaks_ridge(signal,coefficients,neighbor=3,min_length=3,coeffi_threshold=2000): 
    n_scales, n_points = coefficients.shape
    maxindex=np.zeros(coefficients.shape)

    #极大值搜寻
    for i in range(len(coefficients)-1,-1,-1):
        for j in range(1,len(signal)-1):
            if coefficients[i,j]>coefficients[i,j+1] and coefficients[i,j]>coefficients[i,j-1]: 
                maxindex[i,j]=1  
                # print(i,j)
    # print("Already got the maxindex")

    #脊线搜寻策略
    ridges=[] 
    for j in np.where(maxindex[-1] == 1)[0]:  # 找最大尺度的极大值点
            ridge = [[n_scales-1, j]]  # 新开一条脊线，从最后一行开始
            prev_pos = j
            # 逐行往上追踪
            for i in range(n_scales-2, -1, -1):
                # 在 ±neighbor 范围内寻找极大值
                candidates = [k for k in range(max(1, prev_pos-neighbor), min(n_points-1, prev_pos+neighbor+1)) if maxindex[i, k] == 1]
                if candidates:
                    # 如果有多个候选，可以选最接近的
                    next_pos = min(candidates, key=lambda x: abs(x-prev_pos))
                    ridge.append([i, next_pos])
                    prev_pos = next_pos
                else:
                    # ridge.append([i, np.inf])  # 没找到
                    break

            ridges.append(ridge)
           

    #脊线筛选
    filtered_ridges = []
    for ridge in ridges:
        valid_points = sum(1 for _, pos in ridge if pos != np.inf)
        if valid_points >= min_length: #脊线长度筛选
            ridge_coeffs=[]
            for scale_idx,pos_idx in ridge:
                ridge_coeffs.append(coefficients[scale_idx, pos_idx])
                if np.max(ridge_coeffs) > coeffi_threshold: #脊线能量筛选
                    filtered_ridges.append(ridge)
    return  filtered_ridges


# #调用
# peaks_found=find_peaks1(signal,coefficients,10000) #阈值寻峰
# ridges_found=find_peaks_ridge(signal,coefficients,neighbor=3,min_length=3,coeffi_threshold=100) #小波脊线寻峰
# # print(len(ridges_found))
# #后续对接
# #理想输出peak_ridgefound：峰值位置，峰值大小
# peak_ridgefound=[]
# for ridge in ridges_found:
#     min_scale_pos=min(ridge,key=lambda x:x[0])
#     scale_idx,pos_idx=min_scale_pos
#     if np.isfinite(pos_idx) and int(pos_idx) not in peak_ridgefound:  # 去重
#         peak_ridgefound.append(int(pos_idx))
# print(peak_ridgefound)






# #脊线寻峰结果显示
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

# # 原始信号1
# ax1.plot(x, signal, label="Signal")
# ax1.scatter(x[peak_ridgefound], signal[peak_ridgefound], color='red', s=5)
# ax1.legend()

# # 脊线寻峰结果2
# for ridge in ridges_found:
#     scales = [p[0] for p in ridge]
#     positions = [p[1] for p in ridge]
#     positions = np.array(positions, dtype=float)
#     scales = np.array(scales, dtype=float)
#     mask = np.isfinite(positions)
#     positions = positions[mask].astype(int)
#     scales = scales[mask]
#     ax2.scatter(x[positions], scales, color='red', s=2)
# ax2.set_ylabel("Scale")
# ax2.invert_yaxis()

# # 小波系数图3
# ax3.imshow(coefficients,
#            extent=[x.min(), x.max(), scales.max(), scales.min()],
#            cmap='jet', aspect='auto')
# ax3.set_xlabel("x")
# ax3.set_ylabel("Scale")
# ax3.set_title("CWT Coefficients")

# plt.tight_layout()
# plt.show()
