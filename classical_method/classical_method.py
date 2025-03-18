import cv2
import numpy as np
import matplotlib.pyplot as plt

class ClassicalMethod:
  def __init__(self, imagepath):
    self.imagepath = imagepath

    # Load the image
    self.image = cv2.imread(imagepath)

    # Check if the image was loaded successfully
    if self.image is None:
            raise ValueError(f"Error: Could not open or find the image at {imagepath}")
    
    # Convert image to RGB
    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    # Define the sequence of image processing steps
    self.steps = [
            self.Convert2HLS,
            self.ApplyMask,
            self.Convert2Grayscale,
            self.GaussianBlur,
            self.DetectEdges,
            self.RegionOfInterest,
            self.HoughLines,
            self.LaneBoundaries,
            self.FillPolygon
        ]

  def Convert2HLS(self):
    """Convert the image to HLS color space for better lane detection."""
    self.hls = cv2.cvtColor(self.image, cv2.COLOR_RGB2HLS)
    return self.hls

  def ApplyMask(self):
    """Apply a mask to extract yellow and white lane colors."""

    # Define yellow color range
    yellow_lower = np.array([15, 30, 115], dtype="uint8")  
    yellow_upper = np.array([35, 204, 255], dtype="uint8")  
    yellow_mask = cv2.inRange(self.hls, yellow_lower, yellow_upper)

    # Define white color range
    white_lower = np.array([0, 200, 0], dtype="uint8")  
    white_upper = np.array([255, 255, 255], dtype="uint8")  
    white_mask = cv2.inRange(self.hls, white_lower, white_upper)

    # Combine yellow and white masks
    lane_mask = cv2.bitwise_or(yellow_mask, white_mask)
    self.masked_image = cv2.bitwise_and(self.image, self.image, mask=lane_mask)
    return self.masked_image

  def Convert2Grayscale(self):
    """Convert the masked image to grayscale."""
    self.grayscale = cv2.cvtColor(self.masked_image, cv2.COLOR_RGB2GRAY)
    return self.grayscale

  def GaussianBlur(self,filter_size=(5,5), sigma=0):
    """Apply Gaussian Blur to reduce noise in the image."""
    self.blur = cv2.GaussianBlur(self.grayscale, filter_size, sigma)
    return self.blur

  def DetectEdges(self, minVal=50, maxVal=150):
    """Detect edges using the Canny edge detection method."""
    self.edges = cv2.Canny(self.blur, minVal, maxVal)
    return self.edges

  def RegionOfInterest(self):
    """Define and apply a region of interest (ROI) mask."""
    height, width = self.edges.shape[:2]
    roi_vertices = np.array([[(0, height), (0, height*0.25),
                              (width, height*0.25), (width, height)]], dtype=np.int32)
    mask = np.zeros_like(self.edges)
    cv2.fillPoly(mask, [roi_vertices], 255)
    self.roi_edges = cv2.bitwise_and(self.edges, mask)
    return self.roi_edges

  def HoughLines(self):
    """Detect lane lines using the Hough Line Transform."""
    self.lines = cv2.HoughLinesP(self.roi_edges, rho=2, theta=np.pi/180, threshold=100, minLineLength=40, maxLineGap=150)

    # Cater for the case when no lines are detected
    if self.lines is None:
       self.lines = []

    return self.lines

  def LaneBoundaries(self):
    """Determine left and right lane boundaries from detected lines."""
    if self.lines == []: # Check if no lanes were detected
       self.left_lane = None
       self.right_lane = None
       return self.left_lane, self.right_lane

    coordinates = []

    for item in self.lines:
        x1, y1, x2, y2 = item[0]
        coordinates.append((x1, y1, x2, y2))

    # Determine left and right lanes based on x-coordinates
    self.right_lane = max(coordinates, key=lambda item: item[0])
    self.left_lane = min(coordinates, key=lambda item: item[0])

    return self.left_lane, self.right_lane

  def FillPolygon(self):
   """Fill the detected lane area with a polygon."""
   if self.left_lane is None or self.right_lane is None:
      self.output = self.image 
      return self.output # Return the original image as fallback

   mask = np.zeros_like(self.image[:,:,0])
   polygon_points = np.array([[self.left_lane[0],self.left_lane[1]], [self.left_lane[2],self.left_lane[3]],[self.right_lane[0],self.right_lane[1]],[self.right_lane[2],self.right_lane[3]]])
   polygon_points = polygon_points.reshape((-1, 1, 2))
   cv2.fillPoly(mask, [polygon_points], 255)
  
   # Color the lane area
   filled_image = self.image.copy()
   filled_image[mask == 255] = [128, 64, 128]
   self.output = filled_image
   return self.output
  
  def process(self):
      """Execute all image processing steps sequentially."""
      for step in self.steps:
          # Call each processing method
          step()  
      return self.output 
