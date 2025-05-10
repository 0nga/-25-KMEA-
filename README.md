[25-KMEA] Knowledge Management for Ethical Autonomy through Genetic Algorithm-Based Learning in Autonomous Vehicles

# Genetic Algorithm for Ethical Decision-Making Simulation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A project using genetic algorithms to optimize neural networks for ethical decision-making scenarios, simulating moral dilemmas between pedestrians and passengers in autonomous systems.

---

## 📋 Description
This repository implements a genetic algorithm to train a neural network for decision-making in simulated scenarios involving:
- **Ethical dilemmas** between pedestrians and passengers.
- **Fitness optimization** based on altruism, costs, and probabilities.
- **Graph generation** to analyze population evolution and model accuracy.

---

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/repo-name.git
   cd repo-name
   ```
3. Install dependencies
  ```bash
   pip install -r requirements.txt
  ```
## Run the Experiment
  ```bash
  python3 run_experiment.py -g 10 -p 100 -e 0.1 -o outputTest
  ```
-g: Number of generations (default: 10)

-p: Population size (default: 100)

-e: Probability of pedestrian fatalities (e.g., 0.1)

-o: Output folder name

- Generate Plots

   ```bash
   python3 plotAll.py -g 10
   ```
   -g: Number of generations to visualize (must match run_experiment.py settings)

## 📁 Project Structure
  ```bash
.
fixedCode/
    └── run_experiment.py          # Main experiment script (Ethical knob approach)
    ├── Configuration.py           # Hyperparameter configuration
    ├── Individual.py              # Individual definition and genetic operations
    ├── ga_general.py              # Genetic algorithm implementation
    ├── plotAll.py                 # Plot generation and metrics
    ├── requirements.txt           # Dependencies
    ├── grafici                    # Graphs directory
    └── outputTest/                # Results and logs (auto-generated)
Deontological/
    └── run_experiment.py          # Main experiment script (Deontological approach)
    ├── Configuration.py           # Hyperparameter configuration
    ├── Individual.py              # Individual definition and genetic operations
    ├── ga_general.py              # Genetic algorithm implementation
    ├── plotAll.py                 # Plot generation and metrics
    ├── requirements.txt           # Dependencies
    ├── grafici                    # Graphs directory
    └── outputTest/                # Results and logs (auto-generated)
Utilitarian/
    └── run_experiment.py          # Main experiment script (Utilitarian Approach)
    ├── Configuration.py           # Hyperparameter configuration
    ├── Individual.py              # Individual definition and genetic operations
    ├── ga_general.py              # Genetic algorithm implementation
    ├── plotAll.py                 # Plot generation and metrics
    ├── requirements.txt           # Dependencies
    ├── grafici                    # Graphs directory
    └── outputTest/                # Results and logs (auto-generated)
SumoTest

```

## ⚙️ Configurable Parameters
Modify Configuration.py to customize:
- Neural network architecture (layers, activations)
- Mutation and crossover probabilities
- Ethical costs (costPedestrian, costPassengers)
- Evolutionary parameters (e.g., POPULATION_SIZE, MAX_GENERATIONS)

## 📊 Results
Generated plots in grafici/ include:
- Max/Average fitness trends
- Confusion matrices
- ROC curves and metrics (precision, recall, F1-score)
- Altruism evolution in the population

## 📄 License

Add License


```bash
---
**Note**: Customize paths (e.g., `fixedCode/outputTest`) and parameters as needed. For complex scenarios, adjust values in `Configuration.py`.
```
 

   
