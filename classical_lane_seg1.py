import cv2
import numpy as np
import matplotlib.pyplot as plt

class ClassicalMethod:
  def __init__(self, imagepath):
    self.imagepath = imagepath
    self.image = cv2.imread(imagepath)
    if self.image is None:
            raise ValueError(f"Error: Could not open or find the image at {imagepath}")
    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

  def _Convert2HLS(self):
    self.hls = cv2.cvtColor(self.image, cv2.COLOR_RGB2HLS)
    return self.hls

  def _ApplyMask(self):
    yellow_lower = np.array([15, 30, 115], dtype="uint8")  
    yellow_upper = np.array([35, 204, 255], dtype="uint8")  
    yellow_mask = cv2.inRange(self.hls, yellow_lower, yellow_upper)

    white_lower = np.array([0, 200, 0], dtype="uint8")  
    white_upper = np.array([255, 255, 255], dtype="uint8")  
    white_mask = cv2.inRange(self.hls, white_lower, white_upper)

    lane_mask = cv2.bitwise_or(yellow_mask, white_mask)
    self.masked_image = cv2.bitwise_and(self.image, self.image, mask=lane_mask)
    return self.masked_image

  def _Convert2Grayscale(self):
    self.grayscale = cv2.cvtColor(self.masked_image, cv2.COLOR_RGB2GRAY)
    return self.grayscale

  def _GaussianBlur(self,filter_size=(5,5), sigma=0):
    self.blur = cv2.GaussianBlur(self.grayscale, filter_size, sigma)
    return self.blur

  def _DetectEdges(self, minVal=50, maxVal=150):
    self.edges = cv2.Canny(self.blur, minVal, maxVal)
    return self.edges

  def _RegionOfInterest(self):
    height, width = self.edges.shape[:2]
    roi_vertices = np.array([[(0, height), (0, height*0.25),
                              (width, height*0.25), (width, height)]], dtype=np.int32)
    mask = np.zeros_like(self.edges)
    cv2.fillPoly(mask, [roi_vertices], 255)
    self.roi_edges = cv2.bitwise_and(self.edges, mask)
    return self.roi_edges

  def _HoughLines(self):
    self.lines = cv2.HoughLinesP(self.roi_edges, rho=2, theta=np.pi/180, threshold=100, minLineLength=40, maxLineGap=150)
    return self.lines

  def _LaneBoundaries(self):
    coordinates = []

    for item in self.lines:
        x1, y1, x2, y2 = item[0]
        coordinates.append((x1, y1, x2, y2))

    self.right_lane = max(coordinates, key=lambda item: item[0])
    self.left_lane = min(coordinates, key=lambda item: item[0])

    return self.left_lane, self.right_lane

  def _FillPolygon(self):
   mask = np.zeros_like(self.image[:,:,0])
   polygon_points = np.array([[self.left_lane[0],self.left_lane[1]], [self.left_lane[2],self.left_lane[3]],[self.right_lane[0],self.right_lane[1]],[self.right_lane[2],self.right_lane[3]]])
   polygon_points = polygon_points.reshape((-1, 1, 2))
   cv2.fillPoly(mask, [polygon_points], 255)

   filled_image = self.image.copy()
   filled_image[mask == 255] = [128, 64, 128]
   self.output = filled_image
   return self.output
  
  def _GetClassicalOutput(self):
    self._Convert2HLS()
    self._ApplyMask()
    self._Convert2Grayscale()
    self._GaussianBlur()
    self._DetectEdges()
    self._RegionOfInterest()
    self._HoughLines()
    self._LaneBoundaries()
    self._FillPolygon()
    plt.imshow(self.output)   
    plt.show()

ImPath = '000037_10.png'
Im = ClassicalMethod(ImPath)
Im._GetClassicalOutput()

# Im._Convert2HLS()
# Im._ApplyMask()
# Im._Convert2Grayscale()
# Im._GaussianBlur()
# Im._DetectEdges()
# Im._RegionOfInterest()
# Im._HoughLines()
# Im._LaneBoundaries()
# Im._FillPolygon()
# plt.imshow(Im.output)
# plt.show()
