import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

fits_file_path = '/Users/rbose/Work/Filaments/SIN_F08.fits'

def bimodal_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
    return a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) + \
           a2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))

def calculate_curvature(image):
    Hy, Hx = np.gradient(image)
    Hyy, Hyx = np.gradient(Hy)
    Hxy, Hxx = np.gradient(Hx)
    K = (Hxx * Hyy - Hxy**2) / ((1 + Hx**2 + Hy**2)**2)
    H = ((1 + Hx**2)*Hyy + (1 + Hy**2)*Hxx - 2*(Hx*Hy)*(Hxy))/(2*((1 + Hx**2 + Hy**2)**(3/2)))
    discriminant = np.maximum(H**2 - K, 0)
    k1 = H + np.sqrt(discriminant)
    k2 = H - np.sqrt(discriminant)
    theta = 0.5 * np.arctan2(2 * Hxy, Hxx - Hyy)
    phi = np.arctan2(-((Hxx - Hyy) + np.sqrt((Hxx - Hyy)**2 + 4 * Hxy**2)), 2 * Hxy)
    return [Hx, Hy], [Hxx, Hxy, Hyy], [H, K], [k1, k2], theta, phi

def curvature_image(data3d, channels, smooth_length=0, smoother=0):
    k1, k2, phis = [], [], []
    slice_2d = None
    for channel in channels:
        slice_2d = data3d[channel, :, :]
        if smooth_length > 0.:
            smoothed_slice = gaussian_filter(slice_2d, sigma=smooth_length)
        else:
            smoothed_slice = slice_2d
        sharp = slice_2d - (smoother * smoothed_slice)
        _, _, _, KK, _, Phi = calculate_curvature(sharp)
        k1.append(KK[0]); k2.append(KK[1]); phis.append(Phi)
    return slice_2d, np.array(k1), np.array(k2), np.array(phis)

def read_fits(fits_file_path):
    hdul = fits.open(fits_file_path)
    data, header = hdul[0].data, hdul[0].header
    hdul.close()
    return header, data

def detect_fil(k2, nk2, NThreshold=1.0):
    min_nk2 = np.min(nk2[0])
    threshold = (NThreshold*min_nk2)
    mask = (k2 < threshold) & (k2 < 0)
    return mask

def plot_detections(data_2d, k2_slice, mask_slice, Phi_slice, channel_id, fit_params=None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax = axes.flatten()
    phi_deg = np.degrees(Phi_slice) % 180
    m = mask_slice
    phi_data = phi_deg[m]

    im0 = ax[0].imshow(k2_slice, origin='lower', cmap='magma')
    ax[0].set_title(f'k2 Curvature (Ch {channel_id})')
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(m.astype(bool), origin='lower', cmap='gray')
    ax[1].set_title('Filament Mask')
    fig.colorbar(im1, ax=ax[1])

    ax[2].imshow(data_2d, origin='lower', cmap='magma')
    overlay = np.zeros((*m.shape, 4))
    overlay[m] = [1, 1, 1, 0.4]
    ax[2].imshow(overlay, origin='lower')
    ax[2].set_title('Overlay (White=Mask)')

    phi_masked = np.where(m, phi_deg, np.nan)
    im3 = ax[3].imshow(phi_masked, origin='lower', cmap='hsv', vmin=0, vmax=180)
    ax[3].set_title('Phi Map (Masked)')
    fig.colorbar(im3, ax=ax[3])

    ax[4].hist(phi_data, bins=60, range=(0, 180), color='skyblue', alpha=0.7)
    ax[4].set_title('Orientation Histogram')
    ax[4].set_xlabel('Degrees')
    ax[4].set_ylabel('Frequency')

    ax[5].hist(phi_data, bins=60, range=(0, 180), density=True, color='skyblue', alpha=0.6, label='Data')
    if fit_params is not None:
        x_range = np.linspace(0, 180, 200)
        ax[5].plot(x_range, bimodal_gaussian(x_range, *fit_params), 'r-', lw=2, label='Total Fit')
        g1 = fit_params[0] * np.exp(-(x_range - fit_params[1])**2 / (2 * fit_params[2]**2))
        ax[5].plot(x_range, g1, 'g--', alpha=0.8, label='G1')
        g2 = fit_params[3] * np.exp(-(x_range - fit_params[4])**2 / (2 * fit_params[5]**2))
        ax[5].plot(x_range, g2, 'm--', alpha=0.8, label='G2')
    ax[5].set_title('Orientation Fit')
    ax[5].set_xlabel('Degrees')
    ax[5].legend(fontsize='x-small')

    plt.tight_layout()
    plt.show()

def process_channels():
    s, sr, t = 50.0, 4.0, 1.2
    channels = np.arange(440, 473)
    noise_channels = [1]
    _, data3d = read_fits(fits_file_path)
    _, _, nk2, _ = curvature_image(data3d, noise_channels)

    for ch in channels:
        slice_2d, _, k2_batch, phi_batch = curvature_image(data3d, [ch], smooth_length=s, smoother=sr)
        mask_batch = detect_fil(k2_batch, nk2, NThreshold=t)
        phi_deg_masked = np.degrees(phi_batch[0][mask_batch[0]]) % 180
        hist, bin_edges = np.histogram(phi_deg_masked, bins=60, range=(0, 180), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fit_p, m1, m2, peak_distance = None, None, None, None
        try:
            fit_p, _ = curve_fit(bimodal_gaussian, bin_centers, hist, p0=[0.02, 45, 15, 0.02, 135, 15])
            m1, m2 = fit_p[1], fit_p[4]
            a1, a2 = fit_p[0], fit_p[3]
            peak_distance = abs(m2 - m1)
            # Ratio of heights to distance between centers
            ratio_a1_dist = a1 / peak_distance if peak_distance != 0 else 0
            ratio_a2_dist = a2 / peak_distance if peak_distance != 0 else 0
        except:
            pass

        plot_detections(slice_2d, k2_batch[0], mask_batch[0], phi_batch[0], ch, fit_params=fit_p)

        num_fil = np.sum(mask_batch[0])
        total = mask_batch[0].size
        print(f"Ch: {ch} | Filament Ratio: {num_fil/total:.4f} | Fil Pixels: {num_fil} / {total}")

        if m1 is not None:
            print(f"  G1 Mean: {m1:.2f}, SD: {fit_p[2]:.2f} | G2 Mean: {m2:.2f}, SD: {fit_p[5]:.2f}")
            print(f"  >> Distance between peaks: {peak_distance:.1f} deg")
            print(f"  >> Height Ratios: G1/Dist: {ratio_a1_dist:.6f} | G2/Dist: {ratio_a2_dist:.6f}")
        else:
            print(f"  Fit failed for channel {ch}")
        print('-' * 40)

process_channels()
