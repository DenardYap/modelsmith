

FAQs:

1. Why are we concern with 1-bit activations and weights? 

Reason 1: 
You can achieve very fast matrix multiplication between two matrices containing only -1 and 1, because the dot product between the vectors a and b essentially becomes this formula
-> First map -1 (of both a and b) to 0
-> Then do 2 * POPCOUNT( XNOR ( mapped_a, mapped_b ) ) - n, where n is the number of elements in the vector 

This instruction is much faster than the original dot product instruction. 

Reason 2:
Most processors nowadays support 8-bit SIMD instructions, for example, the following C++ intrinsic functions (C++ functions that mapped directly to efficient assembly instructions) perform XNOR between a_vec and b_vec in parallel with 16 8-bit data at a time on a 128-bit register:

uint8x16_t xnor_vec = vmvnq_u8(veorq_u8(a_vec, b_vec));

The resulting vector xnor_vec is a uint8x16_t type, which is basically just an "array" containing 16 8-bit data. 

Now, I am not sure if there exists 1-bit SIMD instructions, but what you can do with 1-bit data is you can first pack them into "fake" 8-bit data, then perform SIMD instructions (like the one above) on these "fake" 8-bit data, which means, you can process up to 128 binary data at once. 

TODO: Give an example

2. How about other-bit quantization for activations and weights? 


3. Why are we implementing True Quantization over Fake Quantization?

In Quantization-Aware Training (QAT), there are two main ways of training the model. Fake Quantization (FQ) is where the quantized model is first trained with high-precision data types (usually float32), then convert to lower-bit data types after training (essentially, FQ QAT is first doing FQ then PTQ after training). The inputs are first quantized into the low-bit (e.g., 8-bit) data type first, then quantize back - this introduces a rounding error in the requantized value, which is proven to be helpful later when converting the model to lower-bit data types. 

Personal Note: The way I understood it is because the model was "aware" of the rounding error, it adapts better to the integer/low-bit data types during training, even though it wasn't trained in the low-bit data types before. Essentially, the distribution of the data is very similar, just shifted. 

True Quantization (TQ) on the other hands, is training the model directly in lower-bit values, like 1-bit, in our case. The main challenge with this approach is that the model has less expressiveness, as everything is either -1 or 1, this means the model should theoretically be harder to converge (this is shown in some papers where binary models' convergences usually take longer TODO: Cite papers). 

Other secondary challenges includes the non-differentiability of functions like signum and requiring larger learning rates during training. 

However, TQ offers advantages such as more accurate representation of what happens during inference time, as the model l


Personal Note: The way I understand machine learning models is that they are just a gigantic if-else block. The simplest form is a linear model, where anything above the line is classified as Yes, and anything below or on the line is No. A.k.a, an if-else block - then we learn about CNN, LSTM, and all the other crazy models like Transformers. In my opinions, all these models are just different ways of writing if-else blocks. Assuming this is true, then 2 values (-1 and 1) are all we need to construct a model. Float32, which is the default data type (As of 2025) in training a machine learning model, is wasteful in my opinions, there's way too many possible values in the float32 set that goes unused. Even for lower-bit data types like float16, int8, int4, or even int2, too many values can gone unused, and therefore a waste of space and potentially computational resources. 

