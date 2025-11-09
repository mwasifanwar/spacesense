import cv2
import numpy as np
from PIL import Image

class ImageProcessor:
    def __init__(self):
        self.config = Config()
    
    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, tuple(self.config.get('data.image_size')))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        return np.transpose(image, (2, 0, 1))
    
    def enhance_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    def remove_background(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        return thresh