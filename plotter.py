import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# ***************** Modern method *****************
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
ax[0].plot(epochs, train_losses, label="Training Loss", color='b', marker='o')

# Plot the validation loss if it exists
if val_losses is not None:
    ax[0].plot(epochs, val_losses, label="Validation Loss", color='r', marker='x')

# Plot the validation IoU if it exists
if val_iou is not None:
    ax[0].plot(epochs, val_iou, label="Validation IoU", color='g', marker='v')

# Add labels and title
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_title('Loss Over Epochs for Modern Method')
ax[0].legend()

# ***************** Classical method *****************
csv_file2 = "iou.csv"
iou_df = pd.read_csv(csv_file2)
image_num = iou_df['Image Number']
iou = iou_df['IoU']

ax[1].scatter(image_num, iou, color='g', marker='v')
ax[1].set_xlabel('Image Number')
ax[1].set_ylabel('IoU')
ax[1].set_title('IoU for all images for Classical Method')


# Show figure
plt.show()

# Save figure 
plt.savefig(output_file)


