import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

channel_index = 450

fits_file_path = '/Users/rbose/Work/Filaments/SIN_F08.fits'
hdul = fits.open(fits_file_path)
# hdul = Header Data Unit List: A FITS file is often structured as a collection of segments called HDUs
# hdul.info(): Provides a summary of how many HDUs are in the file and their dimensions.

data = hdul[0].data
header = hdul[0].header
hdul.close()
'''
hdul[0] refers to the primary HDU, which contains the main data (brightness temp in K of the entire image)
and the metadata (header: details of the telescope used. date of observation, etc)
'''

#print(data.shape) #gives the output: (933, 268, 261): image is 261x268 pixels with 933 frequency channels


slice_2d = data[channel_index, :, :] #represents a 2D cross-section of the 3D data cube

'''
slice_2d contains all spatial data at channel 466
for a range of values slice_2d = data[x1:x2, y1:y2, f1:f2] and then calculate mean intensity
I = np.mean(slice_2d, axis=2), axis=2 means collapsing the frequency channel
'''

# Apply Gaussian smoothing
# sigma represents the standard deviation of the Gaussian kernel
sigma_value, alpha = 70, 3.
smoothed_slice = gaussian_filter(slice_2d, sigma=sigma_value) # data type - numpy array
usm1 = slice_2d - (alpha * smoothed_slice)
usm2 = slice_2d + alpha * (slice_2d - smoothed_slice)

fig, axes = plt.subplots(1, 2, figsize=(13,4))

im0 = axes[0].imshow(slice_2d, origin='lower')
axes[0].set_title('Original 2D image')
fig.colorbar(im0, ax=axes[0])
axes[0].set_xlabel('pixels')
axes[0].set_ylabel('pixels')

im1 = axes[1].imshow(usm1, origin='lower')
axes[1].set_title('USM image')
fig.colorbar(im1, ax=axes[1])
axes[1].set_xlabel('pixels')
axes[1].set_ylabel('pixels')

plt.tight_layout()
plt.show()



