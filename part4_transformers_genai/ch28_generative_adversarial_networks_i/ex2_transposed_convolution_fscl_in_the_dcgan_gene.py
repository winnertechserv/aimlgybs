"""
Chapter 28, Example 2: Transposed Convolution (FSCL) in the DCGAN Generator
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This shows why the generator needs a different operation from the discriminator.
Ordinary convolution shrinks the 4x4 input to a 3x3 feature map - and you can check each
value by hand. We then build the convolution matrix C, check e = Cx, and use C-transpose
to upsample back to 4x4. The result confirms that the shape is restored but the values
are not the original ones.

Run it:           python ex2_transposed_convolution_fscl_in_the_dcgan_gene.py
Expected output:  expected_output/ex2_transposed_convolution_fscl_in_the_dcgan_gene.txt
"""
# Chapter 28 - Example 2: Transposed Convolution (FSCL) used in the DCGAN generator
import numpy as np
np.set_printoptions(suppress=True)

print("=== Transposed Convolution (Up-Convolution / FSCL) ===\n")
print("The DISCRIMINATOR uses ordinary convolution: image -> smaller feature map.")
print("The GENERATOR must go the other way: small input -> larger image.")
print("It uses Transposed Convolution, also called Fractional-Strided")
print("Convolution (stride S < 1), which is why the layer is named FSCL.\n")

# ---------- Ordinary (down) convolution using the convolution matrix ----------
x = np.array([[5, 9, 3, 1],
              [7, 2, 6, 4],
              [1, 8, 5, 2],
              [3, 0, 7, 9]], dtype=float)  # 4x4 input
K = np.array([[0.2, -0.5],
              [0.7, -0.3]], dtype=float)            # 2x2 kernel

def conv2d(img, ker, stride=1):
    kh, kw = ker.shape
    oh = (img.shape[0]-kh)//stride + 1
    ow = (img.shape[1]-kw)//stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(img[i*stride:i*stride+kh, j*stride:j*stride+kw] * ker)
    return out

feat = conv2d(x, K, 1)
print("Input image x (4x4):"); print(x.astype(int))
print("\nKernel K (2x2):"); print(K)
print(f"\nDown-convolution output (3x3)  W2 = (W1-F)/S + 1 = (4-2)/1+1 = 3:")
print(np.round(feat, 2))
print("  -> W2 < W1 : the feature map SHRANK\n")

# ---------- Build the convolution matrix C, then use C^T to upsample ----------
def convolution_matrix(ker, in_h, in_w):
    kh, kw = ker.shape
    oh, ow = in_h-kh+1, in_w-kw+1
    C = np.zeros((oh*ow, in_h*in_w))
    for i in range(oh):
        for j in range(ow):
            row = i*ow + j
            for a in range(kh):
                for b in range(kw):
                    C[row, (i+a)*in_w + (j+b)] = ker[a, b]
    return C

C = convolution_matrix(K, 4, 4)
x_flat = x.reshape(-1)
e_flat = C @ x_flat
print(f"Convolution matrix C shape: {C.shape}  (9 outputs x 16 inputs)")
print(f"Verify  e_bar = C x_bar  matches the direct convolution: "
      f"{np.allclose(e_flat.reshape(3,3), feat)}\n")

# ---------- Transposed convolution: x_tilde = C^T e ----------
print("=== Transposed Convolution: x_tilde_bar = C^T e_bar ===\n")
up_flat = C.T @ e_flat
up = up_flat.reshape(4, 4)
print(f"C^T shape: {C.T.shape}  (16 outputs x 9 inputs) -> UPSAMPLES 3x3 back to 4x4")
print("Result (4x4):")
print(np.round(up, 2))
print(f"\nIs the result equal to the original x? {np.allclose(up, x)}")
print("NO - transposed convolution restores the SHAPE, not the VALUES.")
print("(Note: x_tilde is not the same as x.)\n")

# ---------- Generator in DCGAN ----------
print("=== Generator in DCGAN ===\n")
print("A 100-dim noise vector z is progressively upsampled by FSCL layers:")
stages = [("Unflattened input", "4x4x1024"), ("FSCL 1", "8x8x512"),
          ("FSCL 2", "16x16x256"), ("FSCL 3", "32x32x128"),
          ("FSCL 4 / Conv", "64x64x3  (generated image)")]
for name, shape in stages:
    print(f"  {name:<20} -> {shape}")
print("\nEach FSCL doubles height and width while reducing depth,")
print("turning random noise z into a full-size synthetic image G(z).")
