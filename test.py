import numpy as np

def peak_correction(ridges_found, wl, intensity, window=5):
    peak_ridgefound = []
    true_peaks_idx = []

    for ridge in ridges_found:
        # 找脊线中尺度最小的位置
        min_scale_pos = min(ridge, key=lambda x: x[0])
        scale_idx, pos_idx = min_scale_pos

        if np.isfinite(pos_idx):
            pos_idx = int(pos_idx)
            if pos_idx not in peak_ridgefound:  # 去重
                peak_ridgefound.append(pos_idx)

                # 在原始数据中附近寻找极大值
                left = max(0, pos_idx - window)#避免越界
                right = min(len(intensity) - 1, pos_idx + window)#避免越界
                local_region = intensity[left:right+1]

                local_max_idx = np.argmax(local_region) + left
                true_peaks_idx.append(local_max_idx)

    # 转换成波长和强度
    true_peaks_wl = [wl[i] for i in true_peaks_idx]
    true_peaks_int = [intensity[i] for i in true_peaks_idx]

    return true_peaks_idx, true_peaks_wl, true_peaks_int
