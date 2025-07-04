import torch
import cv2
from torchvision import transforms
from PIL import Image
import time

from framework.processor import Processor
from framework.inference import Inference
from framework.color_map import color_map
from framework.model import DeepLabV3Plus

# Inputs 
MODEL_PATH = "./experiments/bs8_lr0.000146_epochs50/final_model.pth"
input_video_path = 'test-videos/my_recorded_video.mp4'
output_video_path = 'test-videos/my_recorded_video-out.mp4'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 13

# Load the model
model = DeepLabV3Plus(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location = DEVICE))
model.to(DEVICE)
model.eval()

# Define preprocessor, processor and inference classes 
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

processor = Processor()
inference = Inference(model, color_map, DEVICE)

# Open video with OpenCV
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0
inference_times = []
start_time = time.time()

print("Starting video processing")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    # Preprocess
    input_tensor = preprocess(pil_img).unsqueeze(0)  # shape: (1, 3, 256, 256)
    input_tensor = input_tensor.to(DEVICE)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_start = time.time()

    with torch.no_grad():
        output = model(input_tensor)  # shape: (1, num_classes, 256, 256)
        # predicted = torch.argmax(output.squeeze(), dim=0).unsqueeze(0)  # shape: (1, 256, 256)
        predicted = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_end = time.time()

    inference_time = inference_end - inference_start
    inference_times.append(inference_time)

    # Convert to RGB segmentation mask
    rgb_mask = processor.convert_mask_to_rgb(predicted, color_map)

    # Resize mask back to original frame size
    resized_mask = cv2.resize(rgb_mask, (width, height), interpolation=cv2.INTER_NEAREST)

    # Overlay mask onto original frame
    overlay = cv2.addWeighted(frame, 0.5, resized_mask, 0.5, 0)

    out.write(overlay)
    frame_count += 1

    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames...")

if torch.cuda.is_available():
    torch.cuda.synchronize()

end_time = time.time()

# Clean up
cap.release()
out.release()
print("Video processing complete. Saved to:", output_video_path)

# Calculate inference FPS
total_time = end_time - start_time
end_to_end_fps = frame_count / total_time

inference_fps = sum(inference_times)/len(inference_times)

print(f"Processed {frame_count} frames in {total_time:.2f} seconds")
print(f"Original video FPS: {fps:.2f}")
print(f"Inference FPS: {inference_fps:.2f}")
print(f"End-to-end FPS: {end_to_end_fps:.2f}")
