import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D

fits_file_path = '/Users/rbose/Work/Filaments/SIN_F08.fits'

def calculate_curvature(image):

  Hy, Hx = np.gradient(image)
  Hyy, Hyx = np.gradient(Hy)
  Hxy, Hxx = np.gradient(Hx)
  K = (Hxx * Hyy - Hxy**2) / ((1 + Hx**2 + Hy**2)**2)
  H = ((1 + Hx**2)*Hyy + (1 + Hy**2)*Hxx - 2*(Hx*Hy)*(Hxy))/(2*((1 + Hx**2 + Hy**2)**(3/2)))

  discriminant = np.maximum(H**2 - K, 0)
  k1 = H + np.sqrt(discriminant)
  k2 = H - np.sqrt(discriminant)

  return [Hx, Hy], [Hxx, Hxy, Hyy], [H, K], [k1, k2]

def curvature_image(data3d, channels, smooth_length=0, smoother=0):


  k1 = []
  k2 = []

  for channel in channels:
    slice_2d = data3d[channel, :, :]
    if smooth_length > 0.:
      smoothed_slice = gaussian_filter(slice_2d, sigma=smooth_length)
    else:
      smoothed_slice = slice_2d
    #sharp =  slice_2d + 1.5 * (slice_2d - smoothed_slice)
    sharp =  slice_2d - (smoother * smoothed_slice)
    Hi, Hij, HK, KK = calculate_curvature(sharp)
    if smoother > 0.:
      k1.append(KK[0])
      k2.append(KK[1])
    else:
      k1.append(KK[0])
      k2.append(KK[1])

  return  slice_2d,np.array(k1), np.array(k2)

def read_fits(fits_file_path):

  hdul = fits.open(fits_file_path)
  data = hdul[0].data
  header = hdul[0].header
  hdul.close()

  return header, data

def detect_fil(k2, nk2, NThreshold=1.):

  min_nk2 = np.min(nk2[0])
  threshold = (NThreshold*min_nk2)
  mask = (k2 < threshold) & (k2 <0)
  # dtype(mask): boolean array
  return mask

def filament_detect(fits_file_path, channels, nchannels, smooth_length,
                    smoother, NThreshold):

  '''
  channel - frequency of the sliced 2d image for filament detection
  nchannel - frequency of the linefree or noise-free sliced 2d image
  '''

  _, data3d = read_fits(fits_file_path)
  data_2d,k1, k2 = curvature_image(data3d, channels, smooth_length, smoother)
  _, nk1, nk2 = curvature_image(data3d, nchannels)


  return data_2d, k2, detect_fil(k2, nk2, NThreshold)


def plot_detections(data_2d, k2, mask):


  # Create a figure with 3 subplots
  fig, axes = plt.subplots(1, 3, figsize=(18, 5))

  # Plot k2 Curvature
  im0 = axes[0].imshow(k2[0], origin='lower', cmap='magma')
  axes[0].set_title('k2 Curvature (Signal)')
  fig.colorbar(im0, ax=axes[0])

  # Plot SLice_2d
  im1 = axes[1].imshow(data_2d, origin='lower')
  axes[1].set_title('Original Slice_2d')
  fig.colorbar(im1, ax=axes[1])

  # Plot Masked Map
  # Using a binary cmap for the mask
  im2 = axes[2].imshow(mask[0].astype(bool), origin='lower', cmap='gray')
  axes[2].set_title('Filament Mask')
  fig.colorbar(im2, ax=axes[2])

  plt.tight_layout()
  plt.show()

def mult_plot():

  # Define the three specific parameter sets: (smooth_length, smooth_k, threshold)
  parameter_sets = [(70., 3., 1.8), (80, 2.5, 1.5), (90, 2, 1.3)]

  # Reference the data channels
  channels = [450]
  noise_channels = [1]

  for s, sr, t in parameter_sets:
      print(f"\n--- Testing Parameter Set: smooth={s}, smoother={sr}, threshold={t} ---")

      # Run the filament detection function
      data_2d, k2, mask = filament_detect(
          fits_file_path,
          channels,
          noise_channels,s,sr,t
          )

      # Plot the results for this set
      plot_detections(data_2d, k2, mask)

      # Print statistics
      num_fil = np.sum(mask)
      total = mask[0].size
      print(f"Filament Pixels: {num_fil} | Total Pixels: {total} | Ratio: {num_fil/total:.4f}")


mult_plot()

