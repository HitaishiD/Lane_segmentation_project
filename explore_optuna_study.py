import optuna 
import matplotlib.pyplot as plt

study1 = optuna.load_study(study_name="study1", 
                           storage="sqlite:///optuna_study1.db")


# ************* Optimization history ************* 
# trials = study1.trials
# values = [trial.value for trial in trials]
# steps = [trial.number for trial in trials]

# plt.figure(figsize=(10, 6))
# plt.plot(steps, values, marker='o', color='b', label="Optimization History")

# plt.title('Optimization History')
# plt.xlabel('Step (Trial Number)')
# plt.ylabel('Final Validation Loss')
# plt.grid(True)
# plt.legend()

# plt.savefig('optimization_history.png')

# ************************************************* 
