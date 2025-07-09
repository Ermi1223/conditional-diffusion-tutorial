---

# MNIST and CIFAR-10 Conditional Diffusion with Class and Text Conditioning

This project implements a **Conditional Denoising Diffusion Probabilistic Model (DDPM)** on the **MNIST** and **CIFAR-10** dataset using both **class labels** and **text prompts** for conditional image generation. The model is built and trained in **Google Colab**, with visualization tools and advanced experiments such as **interpolation** and **accelerated sampling**.

---

## 📋 Project Overview

* **Dataset**: MNIST (Handwritten digits 0–9)
* * **Dataset**: CIFAR-10 
* **Model**: Conditional U-Net with attention blocks
* **Conditioning**:

  * Class-based (digits 0–9)
  * Text-based (e.g., "the digit five")
* **Scheduler**: Custom cosine beta schedule + DDIM for fast sampling
* **Visualization**: Step-by-step denoising, grid plots, interpolation between digits

---

## 🚀 How to Run in Colab

1️⃣ **Install Required Packages**

```python
!pip install diffusers transformers accelerate matplotlib torchvision
```

2️⃣ **Import Libraries & Set Configuration** (Device, batch size, etc.)

3️⃣ **Prepare MNIST Dataset**

* Grayscale images resized to 32×32
* Tokenize text prompts using CLIP

4️⃣ **Define Conditional Diffusion Model**

* U-Net with class & text conditioning
* Sinusoidal time embeddings

5️⃣ **Implement Diffusion Utilities**

* Forward diffusion (noise addition)
* Reverse sampling (denoising)

6️⃣ **Train the Model**

* Run for \~15 epochs
* Checkpoints every 3 epochs
* Generate images after each epoch

7️⃣ **Generate Samples**

* Class-conditioned generation (digits)
* Text-prompt generation (natural language)

8️⃣ **Visualize Denoising Process**

* Plot generation steps from noise to clean digits

9️⃣ **Prompt Engineering**

* Compare class vs. text conditioning outputs

10️⃣ **Accelerated Sampling**

* Use DDIM for faster generation in fewer steps

11️⃣ **Digit Interpolation**

* Interpolate between digits (e.g., 0 → 9)

---

## Key Features

| Feature                       | Description                                           |
| ----------------------------- | ----------------------------------------------------- |
| ✅ Conditional Sampling        | Supports both digit labels and text prompts           |
| ✅ Forward & Reverse Diffusion | Cosine noise schedule and denoising steps implemented |
| ✅ Visualizations              | Step plots, grid plots, class vs text comparisons     |
| ✅ Prompt Engineering          | Explore different prompt formulations                 |
| ✅ Interpolation               | Visualize smooth transitions between digit classes    |
| ✅ Accelerated Sampling        | Faster generation using DDIM scheduler                |

---

## Notable Modifications for MNIST

* Grayscale (1-channel) images and outputs
* Smaller U-Net for efficiency (64/128/256 channels)
* Text conditioning using **CLIP** embeddings
* Class conditioning using learnable embeddings
* Digit interpolation experiment (new)

---

## Example Outputs

* Digit generation: `sample(model, class_label=5)`
* Text prompt: `sample(model, text_prompt="a handwritten six")`
* Interpolation: `interpolate_digits(model, 0, 9, steps=5)`
* DDIM fast sample: `fast_sample(model, text_prompt="the digit five")`

---

