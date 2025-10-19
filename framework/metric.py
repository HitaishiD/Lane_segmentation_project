# Import required libraries

import numpy as np
import torch
import cv2
from torchvision import transforms
from PIL import Image
import time

# Import custom classes
from framework.processor import Processor
from framework.inference import Inference
from framework.color_map import color_map
from framework.model import DeepLabV3Plus

class Metric:
    def in_lane_from_mask(self,output_mask, lane_color, input_image_path=None): 
        
        
        lane_mask = np.all(output_mask == lane_color, axis=-1)
        

        # Calculate the *horizontal* position of the lane region for each row
        H, W, _ =  output_mask.shape
        bottom_percent = 0.1
        H_start = int(H * (1 - bottom_percent))
        lane_mask_bottom = lane_mask[H_start:H, :]

        # Find the leftmost and rightmost column indices of the detected lane area
        # np.where returns a tuple of arrays, the second being the column indices (x-coords)
        lane_cols = np.where(lane_mask_bottom)
        if len(lane_cols[1]) == 0:
            print("No lane detected in the bottom region. Cannot compute road center.")
            road_center_x = W / 2 # Default to image center if no lane is found
        else:
            # Get the minimum and maximum x-coordinates of the lane pixels
            min_x = np.min(lane_cols[1])
            max_x = np.max(lane_cols[1])
            
            # Estimate the road center as the midpoint of the entire segmented lane area
            road_center_x = (min_x + max_x) / 2

        # --- 2. Vehicle Center Position ---

        # The vehicle center is assumed to be at the center of the image
        # Let the vehicle center be on the x-axis, at the bottom center of the image for simplicity
        vehicle_center_x = W / 2

        print(road_center_x, max_x, vehicle_center_x)

        # --- 3. Position Calculation ---

        if road_center_x <  vehicle_center_x and vehicle_center_x < max_x:
            in_lane = 0
        else:
            in_lane = 1

        # --- 4. Drawing points on the image ---
        if input_image_path is not None: 
            input_img = cv2.imread(input_image_path)
            # Define a Y-coordinate for drawing, near the bottom of the image
            draw_y = H - int(H * 0.05) 

            # 1. Draw Vehicle Center (Red Circle)
                # Vehicle center (red)
            cv2.circle(input_img, (int(vehicle_center_x), draw_y), 6, (0, 0, 255), -1)

            # Road center (yellow)
            cv2.circle(input_img, (int(road_center_x), draw_y), 6, (0, 255, 255), -1)

            # Lane right boundary (red)
            cv2.circle(input_img, (int(max_x), draw_y), 6, (0, 0, 255), -1)
                

            cv2.imwrite("annotated_mask.png", input_img)
        return in_lane

    def in_lane_from_video(self, input_video_path, output_video_path, 
                            MODEL_PATH, DEVICE, NUM_CLASSES): 
        

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
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        # out = cv2.VideoWriter(output_video_path, fourcc, fps, (height, width))  # swapped width and height

        frame_count = 0
        inference_times = []
        start_time = time.time()

        metric = Metric()
        in_lane = []
        lane_col = [k for k,v in color_map.items() if v == 1]

        frames_per_interval = int(fps * 2) if fps > 0 else 60 # Default to 60 frames if fps is zero
        current_frame_pos = 0

        print("Starting video processing")

        while current_frame_pos < total_frames:

            # Set the video read position to the next sample point
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)

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
            

            # check if in lane
            in_lane.append(self.in_lane_from_mask(resized_mask, lane_col))

            out.write(overlay)
            current_frame_pos += frames_per_interval
            frame_count += 1


            if frame_count % 10 == 0:
                print(f"Processed {current_frame_pos} frames...")

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

        # inference_fps = sum(inference_times)/len(inference_times)

        print(f"Processed {current_frame_pos} frames in {total_time:.2f} seconds")
        print(f"Original video FPS: {fps:.2f}")
        # print(f"Inference FPS: {inference_fps:.2f}")
        print(f"End-to-end FPS: {end_to_end_fps:.2f}")
        print(f"In lane result: {in_lane}")


# ------------------ Test in_lane function ---------------
# NUM_CLASSES = 13
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MODEL_PATH = "/home/hitaishi/lane-segmentation/Lane_segmentation_project/experiments/bs8_lr0.000146_epochs31/final_model.pth"
# model = DeepLabV3Plus(NUM_CLASSES)
# model.load_state_dict(torch.load(MODEL_PATH, map_location = DEVICE))
# model.to(DEVICE)
 
# inference = Inference(model, color_map, DEVICE)
# # image_path = "/media/hitaishi/9961-4D68/data_semantics/training/image_2/000025_10.png"
# image_path = "/home/hitaishi/Pictures/Screenshots/Screenshot_20251019_110147.png"

# output_mask = inference.predict_from_image_path(image_path)

# original_image = Image.open(image_path)
# original_size = original_image.size
# output_mask_im = Image.fromarray(output_mask)
# output_mask_im = output_mask_im.resize(original_size)
# output_mask_im.save('overtake.png')

# lane_col = [k for k,v in color_map.items() if v == 1]
# metric = Metric()
# print(metric.in_lane_from_mask(output_mask, lane_col, image_path))

# ------------------ Test in_lane video function ---------------
MODEL_PATH = "./experiments/bs8_lr0.000146_epochs50/final_model.pth"
input_video_path = '/home/hitaishi/lane-segmentation/Lane_segmentation_project/test-videos/ot.mp4'
output_video_path = 'test-videos/ot_out.mp4'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 13

metric = Metric()
metric.in_lane_from_video(input_video_path, output_video_path, 
                            MODEL_PATH, DEVICE, NUM_CLASSES)