import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from sklearn.linear_model import LinearRegression
from Wavelet_peakfinding import wavelet_peak_detection

def extract_Ni_value(name):
    # 文件名格式必须是 Ni + 数字
    # 去掉前两个字符 "Ni"，剩下的数字转 float/int
    try:
        return float(name[2:])   # 例如 "Ni12" -> "12" -> 12.0
    except:
        print(f"警告：无法从文件名解析 Ni 含量：{name}")
        return None
    
folder_path = r'D:\LIBS\ElementDetectation\11.10\Fe-Ni_Spec' #光谱库路径
file_list = glob.glob(os.path.join(folder_path, "*.csv"))
elements_list = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
Ni_contents = [extract_Ni_value(name) for name in elements_list]

all_peaks=[]

for Ni_value,filename in zip(Ni_contents, elements_list):
    # print(f"Ni含量: {contents[0]}, 文件名: {contents[1]}")
    file_path = os.path.join(folder_path, filename + ".csv")  
    # 读取光谱
    df = pd.read_csv(file_path)
    wl = df.iloc[:, 0].values
    intensity = df.iloc[:, 1].values

    # 调用你的寻峰算法
    peak_idx, peak_wl, peak_int = wavelet_peak_detection(wl, intensity, scales=np.arange(1, 11), neighbor=4, min_length=3, coeffi_threshold=1000, window=5)

    # 收集峰位
    for pw in peak_wl:
        all_peaks.append([Ni_value, pw])


print(all_peaks[1])


# # 转成 DataFrame
# peak_df = pd.DataFrame(all_peaks, columns=["Ni", "peak_wl"])

# # Step 2：对峰位二维聚类（不同光谱同一条谱线）
# # 这里用 round(2) 防止少量 shift
# # =====================================
# peak_df["peak_group"] = peak_df["peak_wl"].round(2)

# # =====================================
# # Step 3：对每组峰位做线性拟合：peak_wl = a * Ni + b
# # =====================================
# results = []

# for group, sub in peak_df.groupby("peak_group"):

#     if len(sub) < 3:
#         continue    # 样本太少不拟合

#     x = sub["Ni"].values.reshape(-1, 1)
#     y = sub["peak_wl"].values

#     model = LinearRegression()
#     model.fit(x, y)

#     slope = model.coef_[0]
#     intercept = model.intercept_
#     r2 = model.score(x, y)

#     results.append([group, slope, intercept, r2])


# # 转成 DataFrame
# result_df = pd.DataFrame(results, columns=["peak_wl", "slope", "intercept", "R2"])

# # 按 R² 从大到小排序
# result_df = result_df.sort_values(by="R2", ascending=False)

# print("======= 与 Ni 含量最相关的峰位 =======")
# print(result_df.head(20))

