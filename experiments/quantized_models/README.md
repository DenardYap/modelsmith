# Preface

This repository is the playground for implementing multiple quantized models and benchmarking their performances

Note: this repo uses a different environment, run pip install -r requirements.txt in this repo for your environment

# Tutorial

How do we quantize a matrix? There are many ways to do this, below we describe one such way: 

Given a matrix
[[-0.0053,  0.3793],
[-0.5820, -0.5204],
[-0.2723,  0.1896],
[-0.0140,  0.5607]]

And say we want to quantize it to 8-bit, so *bit_width = 8*. For simplicity we also gonna assume *zero_point = 0*

1. We first need to determine the *scale* of the matrix, we can do so by using the formula.
scale = (max_float - min_float) / (max_int - min_int)

In our case,
max_float = 0.5607
min_float = -0.5820
max_int = 127
max_int = -128

Thus, scale ~ 0.004482.

2. Once we have the scale, we can obtain the integer form of each value by doing 

int = round(float / scale + zero_point)
Thus, the first entry -0.0053 get quantized into round(-0.0053 / 0.004482) = 1

Now, we can directly do backpropagation using the quantized, integer matrix:
[[  -1,   83],
[-127, -114],
[ -59,   41],
[  -3,  122]]

But a more common way is to rescale it back to float again using the scale again. This technique is called fake quantization, although it looks weird at first, by doing so, the model will be "aware" of the quantization (thus it's called Quantization-Aware Training). Then, after the training is done, Post-Training Quantization (PTQ) is performed on the model such that everything will be in int during training. PTQ after QAT will do much better, as the model learned about the integer distribution during training, and all the rounding errors were discarded.

Anyway, to rescale it back:
multiply each entry of 
[[  -1,   83],
[-127, -114],
[ -59,   41],
[  -3,  122]]
by  0.004482
->
[[  -0.004482, 0.372006],
[-0.569214, -0.510948],
[ -0.264438,   0.183762],
[  -0.013446,  0.546804]]
Compare it to the original matrix yourself, you can see they are quite different.
[[-0.0053,  0.3793],
[-0.5820, -0.5204],
[-0.2723,  0.1896],
[-0.0140,  0.5607]]
This is what I meant by learned about integer distribution during training, although the model is trained with the float matrix above.

Note: here we calculate the scale per tensor, which is the 4x2 matrix above, but as mentioned, there are many ways to do this, such as calculating the scales per batch, etc. 
Furthermore, there are multiple formulas for calculating *scale*, the min-max linear quantization above is just one of them.
Brevitas calculathe scale by taking the 99.999 percentile of the absolute value for 300 steps

## Why is zero-point needed?
When your data (float) is not symmetric by default, we need a zero-point. 

Consider a data distribution where the float values range from [-0.5, 1.5], and we want to quantize the data to the range [-128, 127]. Without a zero point, we will obtain the scale like this 
scale = (1.5 - (-0.5)) / (127 - (-128)) = 2/255 ~ 0.00784
Quantizing it, we get:
1.5 -> maps to 
int = round(float / scale + zero_point)
    = round(1.5 / 0.00784 +0)
    = round(191.32) = 191 -> ❌ bad, because we are out of range (max range is 127)
-0.5 -> maps to 
int = round(float / scale + zero_point)
    = round(-0.5 / 0.00784 +0)
    = round(-63.7755) = -64

The other problem is since the data distribution is in [-0.5, 1.5], the center is not 0, it is 0.5. Symmetric quantization like the min-max scale quantization assumes 0.0 is the midpoint of the float range, which is incorrect in this case.
The positive part (0.0 to 1.5) gets mapped to [0 → 191]
The negative part (-0.5 to 0.0) gets mapped to [-64 → 0] 
Only 64 values are available for the negative side.

To fix this, we introduce a zero_point, which can be obtained using this formula:
zero_point = round(-float_min / scale) + q_min 
           = round(- (-0.5) / 0.00784) - 128 
           = 64 - 128 
           = -64

Thus, the zero_point is -64. Now let's redo the int quantization again:

1.5 -> maps to 
int = round(float / scale + zero_point)
    = round(1.5 / 0.00784 - 64)
    = round(127.32) = 127 -> ✅
-0.5 -> maps to 
int = round(float / scale + zero_point)
    = round(-0.5 / 0.00784 - 64)
    = round(-127.77) = -128

Now, the quantized values fully utilize the int8 range, and the float range is evenly mapped without overflow.
Then, we can perform the dequantization as normal:
max value 127 maps to:
float = (int - zero_point) * scale
      = (127 + 64) * 0.00784 
      = 1.49744 -> Original max was 1.5! 
min value -128 maps to:
float = (int - zero_point) * scale
      = (-128 + 64) * 0.00784 
      = -0.50176-> Original min was -0.5! 

Asymmetric quantization with a properly chosen zero point avoids overflow, preserves zero, and ensures full usage of the integer range for skewed float distribution.

TODO: figure out if data is symmetric or asymmetric in linear, relu, conv, etc. If so, what are the zero_points for 
*
You often use symmetric quant for weights
And asymmetric quant for activations
*

## Other ways to calculate scales and their pros and cons

TODO 

## QAT - Training directly in low-bit (true low-bit quantization) vs fake quantization

Above we presented fake quantization, which is a very common way to do QAT. But some researchers also proposed training directly in low-bit (TODO: cite papers). So what are the pros and cons of training directly in low-bit?

Pros:
1. Since the model is trained in low-bit it requires less memory, therefore the training can run on lower-end CPU/GPUs that don't have as much memory.
2. Since the model is directly trained in low-bit integers, it might adapt to the distribution much better than fake quantization. 
3. Custom kernels can be implemented for low-bit integers during training (e.g. low-bit Backpropagation ). This can reduce the training time. How will this be done? Imagine you have two 1-bit matrix, the matrix multiplication between them is simply just POPCOUNT + AND, and it can be implemented super efficiently by leveraging Bit Packing and SIMD.  Meaning custom building blocks like Linear and Conv2D might need to be rewritten to exploit that.

Cons:
1. Harder to converge, although more and more researches have showed that this is possible. For example, it's proven that usually a larger learning rate is required.
2. More complexity, custom kernels and training logic have to be developed, like using STE for gradient discent. 