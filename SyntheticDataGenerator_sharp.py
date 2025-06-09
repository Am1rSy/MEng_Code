import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, util
import random

class SyntheticDataGenerator_sharp:
    def __init__(self, nx=128, ny=128, n_images=1,  min_freq = -600 ,max_freq=200.0, decays=None, seed=None):
        self.nx = nx
        self.ny = ny
        self.n_images = n_images
        self.decays = decays
        self.seed = seed
        self.min_freq = min_freq
        self.max_freq = max_freq
        
        if self.decays is not None and len(self.decays) != self.n_images:
            raise ValueError("Length of decays must match n_images.")

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def generate_field_map(self):
        """
        Generate a random off-resonance field map in Hz, roughly in [-max_freq, max_freq].
        """
        fmap = self.generate_random_smooth_image(decay=random.uniform(90, 100))
        # Scale to +/- max_freq range
        fmap = (self.max_freq - self.min_freq) * fmap + self.min_freq
        return fmap
    def generate_field_map_simple(self, binned_imgs):
        """
        Generate a random off-resonance field map in Hz, roughly in [-max_freq, max_freq].
        """
        fmap_1 = np.where(np.abs(binned_imgs[0])>0.4, 1, 0)
        # Scale to +/- max_freq range
        fmap = (self.max_freq - self.min_freq) * fmap + self.min_freq
        return fmap

    def generate_random_smooth_image(self, decay=0.02):
        """
        Internal helper: generate a 2D random smooth image by weighting
        random k-space data with an exponential decay, then iFFT.
        """
        kspace = (np.random.randn(self.nx, self.ny) +
                  1j * np.random.randn(self.nx, self.ny))

        x = np.linspace(-5, 5, kspace.shape[0])
        y = np.linspace(-5, 5, kspace.shape[1])
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X**2 + Y**2) 
        weighting = np.exp(-decay * (r))
        kspace_weighted = kspace * weighting

        image = np.fft.ifft2(kspace_weighted)
        image_real = np.abs(image)
        image_norm = (image_real - image_real.min()) / (np.ptp(image_real) + 1e-9)
        return image_norm

    def create_composite_image(self):
        """
        Generate complex Synthetic Images (BUGGED)
        """
        # Generate the n_images
        noise = np.random.rand(self.nx, self.ny)
        blurred = filters.gaussian(noise, sigma=random.uniform(5,10))

        binary = blurred > 0.5
        pattern = np.where(binary, 0.0, 1.0)
        # 4. Create a radial fade mask
        cx, cy = self.nx // 2, self.ny // 2
        Y, X = np.ogrid[:self.nx, :self.nx]
        r = np.sqrt((X - cx)**2 + (Y - cy)**2)

        R = self.nx // 2 - 10
        # How wide the "soft" border region should be
        border_width = 20
        
        # Inner radius where the circle is at full intensity
        R_inner = R - border_width
        
        # Initialize a fade array
        fade = np.ones_like(r)
        
        # Outside the main circle => 0
        fade[r >= R] = 0.0
        
        # Smoothly transition in the border zone: [R_inner, R]
        in_border_zone = (r >= R_inner) & (r < R)
        fade[in_border_zone] = 1.0 - (r[in_border_zone] - R_inner) / (R - R_inner)
        background_noise = np.random.normal(loc=0.5, scale=0.1, size=(self.nx, self.ny))
        final_image = (0.6 * pattern + 0.4 * background_noise) * fade
        
        final_image = np.clip(final_image, 0, 1)
        return final_image

    def create_composite_image_simple(self):        
        """
        Generate simple Synthetic Images
        """
        # Generate the n_images
        noise = np.random.rand(self.nx, self.ny)
        blurred = filters.gaussian(noise, sigma=random.uniform(5,10))

        binary = blurred > 0.5
        pattern = np.where(binary, 0.0, 1.0)
        # 4. Create a radial fade mask
        cx, cy = self.nx // 2, self.ny // 2
        Y, X = np.ogrid[:self.nx, :self.nx]
        r = np.sqrt((X - cx)**2 + (Y - cy)**2)

        R = self.nx // 2 - 10
        # How wide the "soft" border region should be
        border_width = 20
        
        # Inner radius where the circle is at full intensity
        R_inner = R - border_width
        
        # Initialize a fade array
        fade = np.ones_like(r)
        
        # Outside the main circle => 0
        fade[r >= R] = 0.0
        
        # Smoothly transition in the border zone: [R_inner, R]
        in_border_zone = (r >= R_inner) & (r < R)
        fade[in_border_zone] = 1.0 - (r[in_border_zone] - R_inner) / (R - R_inner)
        background_noise = np.random.normal(loc=0.5, scale=0.1, size=(self.nx, self.ny))
        final_image = (0.6 * pattern + 0.4 * background_noise) * fade
        
        final_image = np.clip(final_image, 0, 1)
        
        # Initialize a fade array
        fade = np.ones_like(r)
        
        # Outside the main circle => 0
        fade[r >= R] = 0.0
        cropped_pattern = pattern * fade
        fmap_1 = np.where(cropped_pattern >=0.1, cropped_pattern, 0)
        fmap_2 = (pattern ==0) * fade
        fmap = np.zeros((2, final_image.shape[0], final_image.shape[1]), dtype=np.float32)
        fmap[0] = fmap_1
        fmap[1] = fmap_2
        
        binned_imgs = np.zeros((final_image.shape[0], final_image.shape[1], 2), dtype=np.float32)
        binned_imgs[...,0] = np.where(fmap_1 ==1, final_image, 0)
        binned_imgs[...,1] = np.where(fmap_2 ==1, final_image, 0)
        return final_image, binned_imgs,fmap
    
    def create_freq_bin_image(self, composite_img, fmap, num_bins =5):
        
        freq_bins = np.linspace(self.min_freq, self.max_freq, num= num_bins)
        off_resonance = fmap 
        bin_indices = np.digitize(off_resonance, freq_bins) - 1
        
        binned_imgs = np.zeros((composite_img.shape[0], composite_img.shape[1], num_bins), dtype=np.float32)
        for i in range(composite_img.shape[0]):
            for j in range(composite_img.shape[1]):
                # print(i,j)
                b = bin_indices[i,j]
                if b < 0: b = 0
                if b >= num_bins: b = num_bins - 1
                binned_imgs[i,j,b] = composite_img[i,j]  # Put the voxel intensity in that frequency bin
                
        return binned_imgs

    def create_soft_freq_bin_image(self, composite_img, fmap, min_freq, max_freq, num_bins=5):

        H, W = composite_img.shape
    
    
        bin_centers = np.linspace(min_freq, max_freq, num=num_bins)  # e.g. [f0, f1, …, f_{num_bins−1}]
        # The distance between adjacent centres:
        if num_bins > 1:
            bin_width = bin_centers[1] - bin_centers[0]
        else:
            # If you only have one bin, width is irrelevant (all weights=1).
            bin_width = max_freq - min_freq if max_freq != min_freq else 1.0
    
        # Prepare output array:
        binned_imgs = np.zeros((H, W, num_bins), dtype=np.float32)
    
        
        diff = np.abs(fmap[:, :, np.newaxis] - bin_centers[np.newaxis, np.newaxis, :])  # shape (H, W, num_bins)
    
    
        if bin_width > 0:
            weights = np.maximum(0.0, 1.0 - diff / bin_width)  # shape (H, W, num_bins)
        else:
            weights = np.ones_like(diff)
    
        # Normalize along the bin‐axis 
        weight_sums = np.sum(weights, axis=2, keepdims=True)  # shape (H, W, 1)
        # Avoid division by zero: if weight_sums=0, set to 1 so normalized_weights=0.
        weight_sums[weight_sums == 0] = 1.0
        normalized_weights = weights / weight_sums  # shape (H, W, num_bins)
        
        binned_imgs = composite_img[:, :, np.newaxis] * normalized_weights
    
        return binned_imgs, bin_centers

    def create_freq_bin_image_simple(self, composite_img):
        
        fmap_1 = np.where(composite_img>0.4, composite_img, 0)
        fmap_2 = np.where(np.logical_and(composite_img>0, composite_img<=0.3), composite_img, 0)
        
        binned_imgs = np.zeros((composite_img.shape[0], composite_img.shape[1], 2), dtype=np.float32)

        binned_imgs[...,0] = fmap_1
        binned_imgs[...,1] = fmap_2
                
        return binned_imgs

    def save_data(self, filename, composite, freq_map, binned_img):


        np.savez(filename, 
                composite=composite,
                freq_map=freq_map,
                binned_imgs=binned_img)

        print(f"Data saved to '{filename}'")

        