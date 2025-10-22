"""
angular_spectrum_propagation.py

用途：
    读取一张灰度图（或彩色会自动转灰度），假定物理尺寸为 10 mm x 10 mm，
    使用角谱衍射方法将其传播到 z = 2.0 m（2000 mm），并保存传播后的复振幅和强度图。

依赖：
    numpy, scipy, pillow (PIL), matplotlib

用法示例（命令行）：
    python angular_spectrum_propagation.py --input input_image.png --output out_prefix

主要输出（以 out_prefix 为前缀）：
    out_prefix_input_resampled.png   -- 重采样后作为场的入射强度显示
    out_prefix_prop_intensity.png    -- 传播后强度 (|U|^2) 显示（线性或对数）
    out_prefix_prop_amplitude.png    -- 传播后复振幅幅值 |U|
    out_prefix_prop_phase.png        -- 传播后相位（-pi..pi）
    out_prefix_result.npz            -- numpy 存档，包括 arrays: U_in, U_out, intensity, dx, dy, fx, fy

备注：
    - 单位：长度为 mm。波长默认 532 nm = 0.000532 mm。
    - 为了减少周期边界效应，默认会在像的四周进行零填充（pad_factor 默认为 2）。
    - 若需要不同波长或不同 pad_factor，请在参数中调整。
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

def load_and_resample_image(path, N, L=10.0):
    """读取图片，转换为灰度，重采样为 N x N，返回振幅场（实数）和采样间距 dx (mm)。
       L: 物理边长，单位 mm（默认 10 mm）
    """
    img = Image.open(path).convert('L')  # 灰度
    img_resized = img.resize((N, N), resample=Image.BICUBIC)
    arr = np.asarray(img_resized).astype(np.float64)
    # 归一化振幅到 0..1（这里把像素值当作振幅）
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    dx = L / N  # mm
    return arr, dx

def angular_spectrum_propagation(Uin, dx, dy, z_mm, wavelength_mm, pad_factor=2):
    """角谱传播（空间频谱法）
    Uin: 入射复振幅 (2D) (如果只有强度，先以 sqrt(I) 当振幅)
    dx, dy: 采样间距，单位 mm
    z_mm: 传播距离，单位 mm
    wavelength_mm: 波长，单位 mm
    pad_factor: 零填充因子，防止周期卷积效应（取 >=1）；实际FFT尺寸 = pad_factor * N
    返回：Uout（裁剪回原始 N x N）、fx, fy（频率坐标 cycles/mm）
    """
    # 准备复场
    U0 = Uin.astype(np.complex128)
    Ny, Nx = U0.shape
    # zero-pad to reduce wrap-around
    Nx_p = int(pad_factor * Nx)
    Ny_p = int(pad_factor * Ny)
    # make sizes even
    Nx_p += Nx_p % 2
    Ny_p += Ny_p % 2

    # center pad U0 into larger array
    U_pad = np.zeros((Ny_p, Nx_p), dtype=np.complex128)
    y0 = (Ny_p - Ny)//2
    x0 = (Nx_p - Nx)//2
    U_pad[y0:y0+Ny, x0:x0+Nx] = U0

    k = 2.0 * np.pi / wavelength_mm  # wavenumber in mm^-1

    # frequency coordinates (cycles/mm)
    fx = fftshift(fftfreq(Nx_p, d=dx))
    fy = fftshift(fftfreq(Ny_p, d=dy))
    FX, FY = np.meshgrid(fx, fy)

    # spatial frequency squared
    fsq = FX**2 + FY**2

    # construct transfer function H(fx,fy) = exp(i * k * z * sqrt(1 - (lambda*fx)^2 - (lambda*fy)^2))
    # using complex sqrt to allow evanescent components (decay) when argument<0
    arg = 1.0 - (wavelength_mm**2) * fsq
    # use complex sqrt; ensure complex dtype
    sqrt_arg = np.sqrt(arg.astype(np.complex128))
    H = np.exp(1j * k * z_mm * sqrt_arg)

    # forward FFT of padded field
    U_pad_ft = fftshift(fft2(ifftshift(U_pad)))
    # multiply by transfer function
    U_prop_ft = U_pad_ft * H
    # inverse FFT
    U_prop_pad = fftshift(ifft2(ifftshift(U_prop_ft)))

    # crop central region back to original size
    Uout = U_prop_pad[y0:y0+Ny, x0:x0+Nx]

    # return frequency axes for the padded grid (useful for diagnostics)
    # but also return fx_crop, fy_crop matching original sampling region
    return Uout, fx, fy

def save_and_plot_results(Uin, Uout, dx, dy, out_prefix):
    """保存并绘图：输入强度、传播后强度、幅值、相位"""
    intensity_in = np.abs(Uin)**2
    intensity_out = np.abs(Uout)**2
    amplitude_out = np.abs(Uout)
    phase_out = np.angle(Uout)

    # normalize for display
    def norm_display(img):
        a = img.copy()
        a -= a.min()
        if a.max() > 0:
            a /= a.max()
        return a

    # 保存重采样输入图
    plt.figure(figsize=(5,5))
    plt.imshow(norm_display(intensity_in), extent=[-5,5,-5,5], origin='lower', cmap='gray')
    plt.title("Input intensity (resampled) — physical 10 x 10 mm")
    plt.xlabel("x (mm)"); plt.ylabel("y (mm)")
    plt.savefig(out_prefix + "_input_resampled.png", dpi=200, bbox_inches='tight')
    plt.close()

    # 保存传播后强度（线性尺度）
    plt.figure(figsize=(5,5))
    plt.imshow(norm_display(intensity_out), extent=[-5,5,-5,5], origin='lower', cmap='gray')
    plt.title("Propagated intensity at z (linear)")
    plt.xlabel("x (mm)"); plt.ylabel("y (mm)")
    plt.savefig(out_prefix + "_prop_intensity.png", dpi=200, bbox_inches='tight')
    plt.close()

    # 保存传播后 幅值
    plt.figure(figsize=(5,5))
    plt.imshow(norm_display(amplitude_out), extent=[-5,5,-5,5], origin='lower', cmap='gray')
    plt.title("Propagated amplitude |U|")
    plt.xlabel("x (mm)"); plt.ylabel("y (mm)")
    plt.savefig(out_prefix + "_prop_amplitude.png", dpi=200, bbox_inches='tight')
    plt.close()

    # 保存传播后 相位
    plt.figure(figsize=(5,5))
    plt.imshow(phase_out, extent=[-5,5,-5,5], origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi)
    plt.title("Propagated phase (rad)")
    plt.xlabel("x (mm)"); plt.ylabel("y (mm)")
    plt.colorbar(label='radians')
    plt.savefig(out_prefix + "_prop_phase.png", dpi=200, bbox_inches='tight')
    plt.close()

    # 保存 npz
    np.savez(out_prefix + "_result.npz",
             Uin=Uin, Uout=Uout, intensity_in=intensity_in,
             intensity_out=intensity_out, amplitude_out=amplitude_out,
             phase_out=phase_out, dx=dx, dy=dy)

def main():
    parser = argparse.ArgumentParser(description="Angular Spectrum propagation (mm units).")
    parser.add_argument("--input", "-i", type=str, default="input_image.png", help="input image path")
    parser.add_argument("--output", "-o", type=str, default="result", help="output prefix")
    parser.add_argument("--N", type=int, default=1024, help="resample size (N x N)")
    parser.add_argument("--L", type=float, default=10.0, help="physical size in mm (L x L). Default 10 mm")
    parser.add_argument("--z", type=float, default=2000.0, help="propagation distance in mm (default 2000 mm = 2 m)")
    parser.add_argument("--wavelength_nm", type=float, default=532.0, help="wavelength in nm (default 532 nm)")
    parser.add_argument("--pad_factor", type=float, default=2.0, help="zero-pad factor to reduce wrap-around (>=1)")
    args = parser.parse_args()

    in_path = args.input
    out_prefix = args.output
    N = args.N
    L = args.L
    z_mm = args.z
    wavelength_mm = args.wavelength_nm * 1e-6  # nm -> mm
    pad_factor = args.pad_factor

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input image {in_path} not found. Please provide a valid image.")

    # load and resample
    I_resampled, dx = load_and_resample_image(in_path, N, L=L)
    dy = dx

    # assume amplitude = sqrt(I). If original image intended as amplitude, you can skip sqrt.
    Uin = np.sqrt(I_resampled)

    print(f"Image loaded and resampled to {N}x{N}; dx = {dx:.6e} mm. wavelength = {wavelength_mm:.6e} mm.")
    print(f"Propagating to z = {z_mm/1000.0:.3f} m with pad_factor = {pad_factor} ...")

    Uout, fx, fy = angular_spectrum_propagation(Uin, dx, dy, z_mm, wavelength_mm, pad_factor=pad_factor)

    print("Propagation finished. Saving results ...")
    save_and_plot_results(Uin, Uout, dx, dy, out_prefix)
    print("Saved files with prefix:", out_prefix)

if __name__ == "__main__":
    main()
