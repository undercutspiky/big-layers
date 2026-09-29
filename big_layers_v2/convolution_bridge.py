"""The experimental code's original C++ signature, forwarded to the same ATen operator."""
import torch


class _ConvolutionBridge:
    @staticmethod
    def convolution_backward(input, weight, grad_output, stride, padding, output_padding, dilation,
                             groups, benchmark, deterministic, allow_tf32, output_mask):
        return torch.ops.aten.convolution_backward.default(grad_output, input, weight, None, stride,
            padding, dilation, False, output_padding, groups, output_mask)


cpp_conv = _ConvolutionBridge()
