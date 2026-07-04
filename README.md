# Flow Matching Labs

**Practice: Implementation of the flow matching labs**

A comprehensive educational repository containing three progressive Jupyter notebooks that guide you through the theory and implementation of **Flow Matching** and **Score Matching** techniques for generative modeling. These labs build from fundamental concepts to advanced applications in image generation.

## Overview

Flow Matching is a powerful framework for training generative models by learning vector fields that guide samples from a simple distribution to a complex data distribution. This repository implements the theoretical concepts through hands-on exercises and visualizations.

## Lab Structure

### **Lab 1: Simulating ODEs and SDEs**
The foundation of the entire framework.

**Key Topics:**
- Ordinary Differential Equations (ODEs) and Stochastic Differential Equations (SDEs)
- Numerical simulation schemes: Euler and Euler-Maruyama methods
- Brownian Motion and Ornstein-Uhlenbeck processes
- Trajectory visualization and analysis

**What You'll Learn:**
- Understand the mathematical formulation of continuous-time dynamical systems
- Implement numerical solvers for differential equations
- Visualize and interpret stochastic processes
- Work with PyTorch tensors for scientific computing

**Key Implementations:**
```python
class ODE(ABC)
class SDE(ABC)
class EulerSimulator
class EulerMaruyamaSimulator
class BrownianMotion
class OUProcess
```

### **Lab 2: Flow Matching and Score Matching**
Introduction to conditional probability paths and the core flow matching framework.

**Key Topics:**
- Conditional probability paths: p_t(x|z)
- Gaussian conditional probability paths
- Conditional vector fields
- Score functions and their relationship to vector fields
- Visualization of probability path evolution

**What You'll Learn:**
- Design conditional probability paths that transport data
- Compute conditional vector fields analytically
- Understand the connection between flow matching and score-based models
- Visualize how samples evolve along learned paths

**Key Implementations:**
```python
class ConditionalProbabilityPath(ABC)
class GaussianConditionalProbabilityPath
class Alpha, Beta  # Schedule functions
class LinearAlpha, SquareRootBeta
class ConditionalVectorFieldODE
```

**Visualizations in This Lab:**
- Source and target distribution heatmaps
- Probability path evolution over time
- Conditional trajectories with vector field guidance

### **Lab 3: Conditional Generative Models for Images**
Advanced application to image generation with classifier-free guidance.

**Key Topics:**
- Conditional generation on MNIST handwritten digits
- Classifier-free guidance (CFG) for enhanced sample quality
- Diffusion Transformer architecture for image synthesis
- Training conditional flow matching models
- Inference with guidance

**What You'll Learn:**
- Train generative models on real image data
- Implement classifier-free guidance for conditional generation
- Use modern architectures (Diffusion Transformers) for high-dimensional data
- Balance computational efficiency with model quality

**Key Implementations:**
```python
class LabeledSampleable(ABC)
class MNISTSampler
class ConditionalVectorField(ABC)
class CFGVectorFieldODE
class CFGFlowTrainer
```

**Visualizations in This Lab:**
- MNIST samples under conditional probability paths
- Noisy to clean image transitions
- Generated digit samples at different guidance scales

## Requirements

The labs require the following Python packages:

```
torch
torchvision
matplotlib
seaborn
numpy
scipy
sklearn
einops
tqdm
```

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/dralois/FM-Labs.git
cd FM-Labs
```

2. Install dependencies (recommended: use a virtual environment):
```bash
pip install torch torchvision matplotlib seaborn numpy scipy scikit-learn einops tqdm
```

3. Start with Lab 1:
```bash
jupyter notebook lab_one.ipynb
```

4. Progress through the labs sequentially:
   - `lab_one.ipynb` → `lab_two.ipynb` → `lab_three.ipynb`

## Learning Path

### Progression
```
Lab 1: Fundamentals
    ↓
Lab 2: Theory & Implementation
    ↓
Lab 3: Real-World Application
```

Each lab builds on the previous one, recycling and extending implementations:
- Lab 1 introduces basic simulation infrastructure
- Lab 2 reuses `ODE`, `SDE`, and `Simulator` classes, adding probability path abstractions
- Lab 3 extends Lab 2 with MNIST data, conditioning mechanisms, and guidance techniques

## Key Concepts

### Conditional Probability Paths
At the heart of flow matching is the concept of a **conditional probability path** that smoothly transitions samples from a simple distribution to a complex one:

$$p_t(x|z) = \mathcal{N}(x; \alpha_t z, \beta_t^2 I_d)$$

where:
- $z$ is sampled from the target data distribution
- $\alpha_t$ and $\beta_t$ are schedule functions (e.g., $\alpha_t = t$, $\beta_t = \sqrt{1-t}$)
- $t \in [0, 1]$ parameterizes the path

### Conditional Vector Fields
The conditional vector field describes how samples should move along the probability path:

$$u_t(x|z) = \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z + \frac{\dot{\beta}_t}{\beta_t}x$$

This vector field is learned via **conditional flow matching** to enable fast generation.

### Classifier-Free Guidance
A technique that improves sample quality by blending unconditional and conditional predictions:

$$\tilde{u}_t(x|y) = (1-w)u_t(x|\emptyset) + w \cdot u_t(x|y)$$

where $w$ is the guidance scale controlling the strength of conditioning.

## Educational Features

✅ **Progressive Complexity**: Each lab incrementally introduces new concepts  
✅ **Hands-On Exercises**: Implement core functions and verify correctness  
✅ **Rich Visualizations**: See probability paths and trajectories in action  
✅ **Modular Design**: Reusable abstractions across all labs  
✅ **Comments & Documentation**: Clear explanations of mathematical concepts  
✅ **Interactive Exploration**: Easily modify parameters and see results  

## Common Tasks in the Labs

### Implementing a New SDE (Lab 1)
```python
class MyProcess(SDE):
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -self.lambda_ * xt  # Your drift term
    
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(xt) * self.sigma  # Your noise term
```

### Creating a Probability Path (Lab 2)
```python
path = GaussianConditionalProbabilityPath(
    p_data=GaussianMixture.symmetric_2D(nmodes=5, std=1.0, scale=10.0),
    alpha=LinearAlpha(),
    beta=SquareRootBeta()
)
```

### Training on Images (Lab 3)
```python
trainer = CFGFlowTrainer(path=path, eta=0.1)
losses, steps = trainer.train(
    model=your_model,
    num_steps=10000,
    lr=1e-3,
    guidance_scale=7.5
)
```

## Troubleshooting

**CUDA Out of Memory**: Reduce batch sizes in the notebooks
**Slow Training**: The labs use educational-sized models; for production use larger architectures
**Missing Data**: Lab 3 downloads MNIST automatically on first run

## References

The labs implement concepts from:
- Liphardt et al. (2024): *Flow Matching for Generative Modeling*
- Kingma et al. (2021): *Variational Diffusion Models*
- Song et al. (2021): *Score-Based Generative Modeling through Stochastic Differential Equations*

## Authors

Created by: Ezra Erives, Paul Hold, Ron Shechter

For feedback or questions: `ezraerives@gmail.com`, `phold@mit.edu`, `ronsh@mit.edu`

## License

Educational material for learning generative modeling techniques.

---

**Get Started**: Begin with `lab_one.ipynb` to learn the fundamentals of simulating differential equations, then progress through the labs to build complete generative models!
