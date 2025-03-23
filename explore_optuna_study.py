import optuna 
import matplotlib.pyplot as plt
import numpy as np

study1 = optuna.load_study(study_name="study2", 
                           storage="sqlite:///optuna_study2.db")

# ************* Best parameters ************* 
print("Best parameters:", study1.best_params)

# ************* Parameters from trial n ************* 
# trial_2 = study1.trials[2]
# print(f"Parameters for Trial 2: {trial_2.params}")

# # ************* Optimization history ************* 
trials = study1.trials
values = [trial.value for trial in trials]
steps = [trial.number for trial in trials]

plt.figure(figsize=(10, 6))
plt.plot(steps, values, marker='o', color='b', label="Optimization History")

plt.title('Optimization History')
plt.xlabel('Step (Trial Number)')
plt.ylabel('Final Validation Loss')
plt.grid(True)
plt.legend()

plt.savefig('optimization_history2.png')

# # *************** Slice plots ********************
param_names = list(study1.best_trial.params.keys())


num_params = len(param_names)
cols = 3  
rows = (num_params // cols) + (num_params % cols > 0) 


fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
axes = axes.flatten()


for i, hyperparameter_name in enumerate(param_names):
    x_values = []  # Hyperparameter values
    y_values = []  # Objective values
    
    # For each trial, extract the relevant hyperparameter and the objective value
    for trial in study1.trials:
        if hyperparameter_name in trial.params:
            x_values.append(trial.params[hyperparameter_name])
            y_values.append(trial.value)
    
    # Sort values
    sorted_indices = np.argsort(x_values)
    x_values_sorted = np.array(x_values)[sorted_indices]
    y_values_sorted = np.array(y_values)[sorted_indices]
    
    # Create the slice plot for the current hyperparameter on the corresponding axis
    ax = axes[i]
    ax.plot(x_values_sorted, y_values_sorted, marker='o', color='b')
    
    ax.set_title(f'Slice Plot for {hyperparameter_name}')
    ax.set_xlabel(hyperparameter_name)
    ax.set_ylabel('Final Validation Loss')
    ax.grid(True)

plt.savefig('slice_plots2.png')