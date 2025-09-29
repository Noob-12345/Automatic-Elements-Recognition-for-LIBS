import os
import glob
import pandas as pd
import numpy as np
import pywt

def wavelet_peak_detection(signal, wl, wavelet='mexh', scales=np.arange(1, 11), 
                           neighbor=4, min_length=3, coeffi_threshold=1000, window=5):

    # ====== 1. 小波变换 ======
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet)
    n_scales, n_points = coefficients.shape
    
    # ====== 2. 极大值点搜索 ======
    maxindex = np.zeros_like(coefficients, dtype=int)
    for i in range(n_scales-1, -1, -1):
        for j in range(1, n_points-1):
            if coefficients[i, j] > coefficients[i, j+1] and coefficients[i, j] > coefficients[i, j-1]:
                maxindex[i, j] = 1
    
    # ====== 3. 脊线跟踪 ======
    ridges = []
    for j in np.where(maxindex[-1] == 1)[0]:  # 从最大尺度的极大值点出发
        ridge = [[n_scales-1, j]]
        prev_pos = j
        for i in range(n_scales-2, -1, -1):
            candidates = [k for k in range(max(1, prev_pos-neighbor),
                                           min(n_points-1, prev_pos+neighbor+1))
                          if maxindex[i, k] == 1]
            if candidates:
                next_pos = min(candidates, key=lambda x: abs(x-prev_pos))
                ridge.append([i, next_pos])
                prev_pos = next_pos
            else:
                break
        ridges.append(ridge)
    
    # ====== 4. 脊线筛选（长度+能量） ======
    filtered_ridges = []
    for ridge in ridges:
        valid_points = sum(1 for _, pos in ridge if pos != np.inf)
        if valid_points >= min_length:
            ridge_coeffs = [coefficients[s, p] for s, p in ridge]
            if np.max(ridge_coeffs) > coeffi_threshold:
                filtered_ridges.append(ridge)
    
    # ====== 5. 峰值矫正 ======
    peak_ridgefound = []
    true_peaks_idx = []
    for ridge in filtered_ridges:
        min_scale_pos = min(ridge, key=lambda x: x[0])  # 最小尺度点
        scale_idx, pos_idx = min_scale_pos
        if np.isfinite(pos_idx):
            pos_idx = int(pos_idx)
            if pos_idx not in peak_ridgefound:
                peak_ridgefound.append(pos_idx)
                left = max(0, pos_idx - window)
                right = min(len(signal)-1, pos_idx + window)
                local_region = signal[left:right+1]
                local_max_idx = np.argmax(local_region) + left
                true_peaks_idx.append(local_max_idx)
    
    # ====== 6. 转换成波长和强度 ======
    true_peaks_wl = [wl[i] for i in true_peaks_idx]
    true_peaks_int = [signal[i] for i in true_peaks_idx]
    true_peaks_wl=np.array(true_peaks_wl)
    return true_peaks_idx, true_peaks_wl, true_peaks_int
