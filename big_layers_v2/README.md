# Experimental hot-potato Big Layers

This directory contains an experimental follow-up implementation. **The results in the paper do not use this code.**
The paper path is `big_layers/`.

## Motivation

The layer-wise implementation in `big_layers/` keeps large intermediate tensors in CPU RAM. A Big Layer copies the
chunk it needs to the GPU, computes the result, and copies the output back to RAM. If the *next* Big Layer could
already fit that complete output on the GPU, immediately copying the result back to RAM and then sending it to the
GPU again wastes transfer time.

The hot-potato idea is therefore:

1. If a Big Layer's complete output fits on the GPU, keep that output on the GPU for the next layer.
2. Save the CPU copy that will be needed later for backward, but do not force the next forward layer to reload the
   same tensor from CPU.
3. Apply the corresponding idea during backward: keep the current gradient/data on the GPU when the preceding
   backward operation can consume it directly, while preserving the host-side state needed by the rest of the graph.

For the input sizes where it works, this can roughly halve the avoidable CPU-to-GPU/GPU-to-CPU movement across a
boundary between consecutive Big Layers.

## Why it is experimental

The optimization requires neighbouring layers to cooperate: a layer has to know whether it received a host tensor or
a GPU-resident hot-potato tensor, whether the next operation can keep it on the GPU, and which host copy must be
retained for backward. That is why this directory contains fused/grouped layer implementations rather than simple
replacements for isolated operators.

During experiments, some combinations of input size and number of Big Layers could stall: the process remained alive
but stopped doing any compute. Reducing the CPU thread count avoided the stall in those cases, but the lower thread
count also made the overall run slower. The practical workaround was to choose the number of Big Layers carefully for 
each input size.

That unresolved stall/threading interaction is the main reason this implementation was not used for the reported
experiments. A robust solution would make this path preferable because it reduces transfers without changing the
underlying Big-Layer idea.

## Contents

The files here explore fused ResNet and ViT blocks, cached GPU intermediates, and the bookkeeping needed to pass
GPU-resident tensors between cooperating layers. They are provided as experimental code for anyone interested in
solving the stalling problem or developing the fused approach further.
