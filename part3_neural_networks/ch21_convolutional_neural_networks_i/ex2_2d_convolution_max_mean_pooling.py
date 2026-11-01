"""
Chapter 21, Example 2: 2D Convolution + Max/Mean Pooling
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Here we build 2-D convolution and both pooling types from scratch, using a small 4x4
image and a 2x2 kernel. You can check every output value, and the max-pooled and mean-
pooled values, by hand. This shows how a feature map is built and then downsampled - and
that pooling has no learnable parameters.

Run it:           python ex2_2d_convolution_max_mean_pooling.py
Expected output:  expected_output/ex2_2d_convolution_max_mean_pooling.txt
"""
# Chapter 21 - Example 2: 2D Convolution + Max/Mean Pooling from scratch
import numpy as np

def conv2d(img, kernel, stride=1):
    kh, kw = kernel.shape
    H, W = img.shape
    out_h = (H - kh)//stride + 1
    out_w = (W - kw)//stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = img[i*stride:i*stride+kh, j*stride:j*stride+kw]
            out[i, j] = np.sum(region * kernel)
    return out

def pool2d(fmap, size=2, stride=2, mode='max'):
    H, W = fmap.shape
    out_h = (H - size)//stride + 1
    out_w = (W - size)//stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = fmap[i*stride:i*stride+size, j*stride:j*stride+size]
            out[i, j] = region.max() if mode == 'max' else region.mean()
    return out

# Input (4x4), 2x2 kernel, stride 1
img = np.array([[5, 9, 3, 1],
                [7, 2, 6, 4],
                [1, 8, 5, 2],
                [3, 0, 7, 9]], dtype=float)
kernel = np.array([[0.2, -0.5],
                   [0.7, -0.3]], dtype=float)

print("Input image (4x4):")
print(img.astype(int))
print("\n2x2 Kernel:")
print(kernel)

fmap = conv2d(img, kernel, stride=1)
print("\nFeature map after 2D convolution (stride=1, 3x3):")
print(np.round(fmap, 2))

# Now pooling on a sample 4x4 feature map
feat = np.array([[ 0.8, -0.1,  3.1, -1.6],
                 [-1.3,  1.5,  2.1, -0.6],
                 [-1.7, -3.0,  2.2,  0.4],
                 [ 2.6,  0.8, -1.5,  2.8]], dtype=float)
print("\n--- Pooling demo on a 4x4 feature map ---")
print("Features before pooling:")
print(feat)
print("\nMax pooling (2x2, stride 2):")
print(np.round(pool2d(feat, 2, 2, 'max'), 3))
print("Expected: [[1.5, 3.1], [2.6, 2.8]]")
print("\nMean pooling (2x2, stride 2):")
print(np.round(pool2d(feat, 2, 2, 'mean'), 3))
print("Expected: [[0.225, 0.75], [-0.325, 0.975]]")
print("\nPooling kernels have NO learnable parameters.")
