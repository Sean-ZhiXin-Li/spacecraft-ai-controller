#  Project Log Day 5 - Imitation Learning: Neural Network Controller Training

 **Date:** 2025-07-21  
 **Author:** Zhixin Li (Sean)  
 **Files:** `train_imitation.py`, `expert_dataset_*.npy`, `imitation_policy_model.joblib`

---

##  Goals for Today

- [x] Load and combine expert controller datasets (`.npy` files)
- [x] Split data into training and test sets (X = state, y = thrust)
- [x] Build and train a neural network model (MLPRegressor)
- [x] Visualize true vs. predicted thrust vectors on the position plane

---

##  What I Accomplished

### 1.  Fully Understood the `envs/OrbitEnv` Environment

- I **successfully reviewed and annotated every single line** in the `OrbitEnv` environment.
- This includes understanding how `reset()`, `step()`, `_get_obs()`, and the `reward()` function work.
- I now fully understand:
  - `Box()` space and its bounds
  - `np.clip()` usage for action limits
  - What each observation/state variable means
  - How orbit radius and termination conditions are calculated
- I now feel confident that I can write a custom Gym-style orbit environment from scratch if needed.

### 2.  First Time Learning MLPRegressor

- Today was my **first-ever encounter** with `sklearn.neural_network.MLPRegressor`.
- I studied all its major parameters:
  - `hidden_layer_sizes`, `activation='tanh'`, `solver='adam'`, `max_iter`
- I understood the basic architecture:
  - Input → Hidden Layers → Output
- Though I’m a complete beginner in neural networks, I made good progress in understanding how it learns to approximate a thrust policy.

### 3. Visualized Thrust Predictions

- I used `matplotlib.pyplot.quiver` to plot:
  - True thrust vectors (green)
  - Predicted vectors by the neural network (red)
- The vectors are anchored at real 2D positions from the test data.
- Since thrust vectors are small in magnitude, I applied a scaling factor (`1e11`) to make them visible.
- The plot gives an intuitive comparison between expert and learned behavior.

---

##  Issues I Faced

###  Dataset Not Found at First
- My initial dataset folder path was incorrect.
- I manually corrected it to `data/data/dataset/`, which fixed the issue.

###  Arrows Too Short to See
- Thrust vector arrows were visually invisible.
- Solved this by adding a `scale_factor` when plotting the quiver plot.

###  No Prior Knowledge of Neural Networks
- I was confused by the structure and purpose of MLPRegressor.
- After reading documentation and seeing the input/output structure, I gained a basic understanding of its functionality.

---

##  Self-Reflection

Today’s work was both **challenging and rewarding**:
- I **fully understood the logic behind the OrbitEnv environment** and added my own comments to each part.
- Although I had **zero prior experience in neural networks**, I managed to **train an MLP-based thrust controller**.
- I also implemented a **clear and meaningful visualization** comparing real and predicted actions.

  I will **focus on understanding the learning mechanics of the neural network** in greater depth, including:
- how `tanh` activation works,
- how weights and biases affect predictions,
- and how training error decreases across iterations.

---

##  Attached Artifacts

- `train_imitation.py` – Neural network-based imitation controller training script
- `imitation_policy_model.joblib` – Saved model after training
- `expert_dataset_*.npy` – Expert demonstration data
- `thrust_quiver_plot.png` – Visualization: predicted vs. true thrust vectors

