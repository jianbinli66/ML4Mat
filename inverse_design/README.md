# Material Design Framework with Invertible Neural Networks

A comprehensive deep learning framework for inverse materials design using invertible neural networks (INNs) that enables bidirectional mapping between material compositions/processing conditions and material properties.

## Overview

This framework implements state-of-the-art invertible neural networks to solve the challenging problem of inverse materials design. Unlike traditional approaches that require separate models for forward prediction and inverse design, our INN-based approach learns a single bidirectional mapping between material features (formula components and processing conditions) and material properties.

Key capabilities:
- **Forward prediction**: Predict material properties from composition and processing conditions
- **Inverse design**: Generate material compositions and processing conditions for desired properties
- **Uncertainty quantification**: Generate multiple valid designs for target properties
- **Hybrid data handling**: Specialized treatment of continuous formula components and binary processing conditions

## Technical Approach

### Invertible Neural Networks

The framework uses affine coupling layer-based INNs that provide exact invertibility while maintaining expressive power. This architecture enables:

1. **Bijective mapping**: Exact reconstruction in both directions without information loss
2. **Latent space regularization**: Gaussian prior in latent space enables sampling and uncertainty estimation
3. **Conditional generation**: Generate diverse solutions for target properties

### Custom Loss Function

We implement a specialized hybrid loss function that appropriately handles different data types:

- **Mean Squared Error (MSE)**: For continuous formula components and properties
- **Binary Cross Entropy (BCE)**: For binary processing conditions with straight-through estimators
- **Consistency regularization**: Joint training of forward and inverse models

### Architecture Components

1. **Forward Model**: Standard feedforward network for property prediction
2. **INN Model**: Invertible network for bidirectional mapping with affine coupling layers
3. **Feature Processing**: Separate handling of formula (continuous) and condition (binary) components

