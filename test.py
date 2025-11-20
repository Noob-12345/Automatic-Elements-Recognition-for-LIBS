import numpy as np
kB=8.617330350e-5

def Boltzmann_fit_iterative(
        I, wl, A, g, E,
        R2_threshold=1e-1,
        R2_start_threshold=0.97,
        max_iter=20,
        verbose=False):
    """
    迭代 Boltzmann 拟合：
    - 初次拟合 R² 高于设定阈值 → 直接返回
    - 初次拟合 R² 低于阈值 → 开启迭代，每次删除偏离最大的点
    - 返回拟合参数和最终 R² 以及 y_full 和 y_used
    """

    # 计算 y
    y = np.log(I * wl / (g * A))  # full y
    idx = np.arange(len(E))

    # ---- 初次拟合 ----
    slope, intercept = np.polyfit(E[idx], y[idx], 1)
    y_pred = slope * E[idx] + intercept

    ss_res = np.sum((y[idx] - y_pred)**2)
    ss_tot = np.sum((y[idx] - np.mean(y[idx]))**2)
    R2_init = 1 - ss_res / ss_tot

    if verbose:
        print(f"Initial fit: R2 = {R2_init:.5f}")
        
    if R2_init >= R2_start_threshold:
        if verbose:
            print("R² 达到要求，不需要迭代")
        T = -1 / (slope * kB)

        # 返回 y_full 和 y_used（初次拟合使用的是全部点）
        return slope, intercept, T, R2_init, y, y

    # ---- 启动迭代：每次删除最大偏差点 ----
    R2_prev = R2_init

    for it in range(max_iter):

        # 残差
        y_pred = slope * E[idx] + intercept
        residuals = y[idx] - y_pred

        # 删除偏离最大的点
        worst = np.argmax(np.abs(residuals))
        worst_idx = idx[worst]

        if verbose:
            print(f"Iter {it+1}: remove index={worst_idx}, residual={residuals[worst]:.5f}")

        idx = np.delete(idx, worst)

        if len(idx) < 2:
            if verbose:
                print("剩余点少于 2 个，停止迭代")
            break

        # 重新拟合
        slope, intercept = np.polyfit(E[idx], y[idx], 1)
        y_pred = slope * E[idx] + intercept

        # 新 R²
        ss_res = np.sum((y[idx] - y_pred)**2)
        ss_tot = np.sum((y[idx] - np.mean(y[idx]))**2)
        R2_new = 1 - ss_res / ss_tot
        
        delta_R2 = abs(R2_new - R2_prev)

        if verbose:
            print(f"    New R2 = {R2_new:.5f}, ΔR2={delta_R2:.6f}")

        # 收敛判定
        if delta_R2 < R2_threshold:
            if verbose:
                print("R² 收敛，停止迭代")
            break

        R2_prev = R2_new

    # 最终温度
    T = -1 / (slope * kB)

    # y_full = y
    # y_used = y[idx]
    return slope, intercept, T, R2_prev, y, y[idx]


