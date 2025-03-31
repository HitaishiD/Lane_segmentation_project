import pandas as pd
import matplotlib.pyplot as plt


# Path to CSV file containing training losses
csv_file = "experiments/bs8_lr0.000146_epochs31/training_losses.csv"
# Path at which to save the plot as an image
output_file = "experiments/bs8_lr0.000146_epochs31/loss_plot.png"


# Read the CSV file into a pandas DataFrame
loss_df = pd.read_csv(csv_file)

# Extract the epoch and loss columns
epochs = loss_df['Epoch']
train_losses = loss_df['Training Loss']
val_losses = loss_df['Validation Loss'] if 'Validation Loss' in loss_df.columns else None
val_iou = loss_df['Validation IoU'] if 'Validation IoU' in loss_df.columns else None

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label="Training Loss", color='b', marker='o')

# Plot the validation loss if it exists
if val_losses is not None:
    plt.plot(epochs, val_losses, label="Validation Loss", color='r', marker='x')

# Plot the validation IoU if it exists
if val_iou is not None:
    plt.plot(epochs, val_iou, label="Validation IoU", color='g', marker='v')

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()

# Show figure
plt.show()

# Save figure 
plt.savefig(output_file)


