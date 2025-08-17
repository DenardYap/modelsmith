import torch
from brevitas.nn import QuantLinear

torch.manual_seed(0)

quant_linear = QuantLinear(2, 4, bias=True, bit_width=4, weight_bit_width=2)

print(f"Original float weight tensor:\n {quant_linear.weight} \n")
print(f"Quantized weight QuantTensor:\n {quant_linear.quant_weight()} \n")
print(f"Quantized weight QuantTensor:\n {quant_linear.quant_weight().value} \n")
print(f"Quantized weight QuantTensor:\n {quant_linear.quant_weight().int()} \n")
print("Bit width:", quant_linear.weight_quant.bit_width())

