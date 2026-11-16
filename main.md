# Gradient Norm

The **gradient norm** represents the magnitude (the size or length) of the gradient vector.

Monitoring and controlling this value during neural network training is crucial: if the magnitude is too high (a phenomenon known as *exploding gradients*), the model's weights will be updated with excessively large steps. This causes the optimization algorithm to overshoot the optimal point, preventing the model from converging properly.

# Norm Types

The two most common ways to calculate the norm of a gradient vector $x$ are:

* **$L_1$ Norm (Manhattan Norm):** The sum of the absolute values of all elements in the vector.

  $$
  \Vert x \Vert_1 = \sum_{i} \vert x_i \vert
  $$

* **$L_2$ Norm (Euclidean Norm):** The square root of the sum of the squares of all elements in the vector (the most commonly used in practice for *clipping*).

  $$
  \Vert x \Vert_2 = \sqrt{\sum_{i} x_i^2}
  $$

# Gradient Clipping

*Gradient Clipping* is a stabilization technique. After calculating the gradient norm, we define a maximum threshold that it is allowed to reach. If the calculated norm exceeds this threshold, we rescale the entire gradient vector to bring it back to a safe size, preserving its original direction while reducing its magnitude.

# Math Logic

When the norm exceeds the defined threshold, the rescaling follows this logic:

* **Gradient Norm ($\Vert g \Vert$):** Calculated according to the chosen norm type ($L_1$ or $L_2$).

* **Scale Factor ($\text{Scalar}$):** The ratio between the maximum allowed threshold and the current gradient norm:

  $$
  \text{Scalar} = \frac{\text{Grad Threshold}}{\text{Grad Norm}}
  $$

* **Scaled Vector ($g_{\text{new}}$):** The original vector multiplied by the scale factor (proportionally reducing its size so that its new norm is exactly equal to the $\text{Grad Threshold}$):

  $$
  \text{Scaled Vector} = \text{Vector} \times \text{Scalar}
  $$

  *(Note: If you choose to divide by the scalar in your implementation, the factor formula must be inverted to $\text{Scalar} = \frac{\text{Grad Norm}}{\text{Grad Threshold}}$, resulting in $\text{Scaled Vector} = \frac{\text{Vector}}{\text{Scalar}}$).*