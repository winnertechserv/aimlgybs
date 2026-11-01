"""
Chapter 21, Example 1: 1D Convolution from Scratch
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This works through a small 1-D convolution by hand. We slide a length-4 kernel across a
10-element signal with stride 2, doing the shift-and-multiply (dot product) at each
stop. The output is the feature map [-0.6, 3.2, 2.8, -0.5], and we confirm the output-
size formula. Note there is no bias and no activation - that is the raw convolution.

Run it:           python ex1_1d_convolution_from_scratch.py
Expected output:  expected_output/ex1_1d_convolution_from_scratch.txt
"""
# Chapter 21 - Example 1: 1D Convolution from scratch
import numpy as np

def conv1d(data, kernel, stride):
    K = len(kernel)
    out = []
    positions = []
    for start in range(0, len(data) - K + 1, stride):
        window = data[start:start+K]
        val = np.dot(window, kernel)
        out.append(round(float(val), 4))
        positions.append(start)
    return out, positions

# A small worked example
data   = np.array([2, 8, 6, 4, 7, 3, 5, 2, 1, 4], dtype=float)
kernel = np.array([0.2, -0.5, 0.7, -0.3], dtype=float)
stride = 2

print("Input data  :", data.astype(int).tolist())
print("Kernel      :", kernel.tolist())
print("Stride      :", stride, "| Kernel size:", len(kernel))
print()

features, pos = conv1d(data, kernel, stride)
print("Step-by-step (shift-and-multiply):")
for f, p in zip(features, pos):
    window = data[p:p+len(kernel)].astype(int).tolist()
    terms  = " + ".join(f"{k}*{w}" for k, w in zip(kernel.tolist(), window))
    print(f"  pos {p}: {terms} = {f}")
print()
print("Feature map extracted:", features)
print("Expected by hand     : [-0.6, 3.2, 2.8, -0.5]")

# Output length formula
W1 = len(data); F = len(kernel); S = stride
W2 = (W1 - F)//S + 1
print(f"\nOutput size formula: (W1 - F)/S + 1 = ({W1} - {F})/{S} + 1 = {W2}")
print("Note: convolution layer has NO bias and NO activation in this raw operation.")
