import pandas as pd
import matplotlib.pyplot as plt



csv_file = "/home/ubuntu/computer-vision/computer-vision/experiments/bs2_lr0.001_epochs10/training_losses.csv"
# Path to your CSV file
#csv_file = 'path_to_your_csv_file/training_losses.csv'

# Read the CSV file into a pandas DataFrame
loss_df = pd.read_csv(csv_file)

# Extract the epoch and loss columns
epochs = loss_df['Epoch']
train_losses = loss_df['Training Loss']
val_losses = loss_df['Validation Loss'] if 'Validation Loss' in loss_df.columns else None

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label="Training Loss", color='b', marker='o')

# Plot the validation loss if it exists
if val_losses is not None:
    plt.plot(epochs, val_losses, label="Validation Loss", color='r', marker='x')

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()



# Save the plot as an image
output_file = "/home/ubuntu/computer-vision/computer-vision/experiments/bs2_lr0.001_epochs10/loss_plot.png"
plt.savefig(output_file)

# Optionally, print the path where the plot is saved
print(f"Plot saved as {output_file}")

