import ffmpeg
import logging
import sys, getopt
import cv2
import numpy as np

class Main(object):

    def __init__(self):
        print("template_matcher (nortoh) v1.0")
        self.execute()

    def execute(self):
        self.video = 'video.mp4'
        self.template_image_file = 'template.png'
        self.template = cv2.imread(self.template_image_file)
        self.template_width, self.template_height = self.template.shape[::1]

        cam = cv2.VideoCapture(self.video)
        
        # Stick to a small section of frames for debugging
        current_frame = 0
        max_frame = 10

        while(current_frame < max_frame):
            ret, image = cam.read()

            if ret:
                # We have a good frame
                
                # Check frame
                self.check_frame(current_frame, image)
                current_frame += 1
            else:
                # We have a bad frame
                break

        cam.release()
        cv2.destroyAllWindows()

    def check_frame(self, current_frame, image):
        
        # print(f'current_frame: {str(current_frame)} frame: {image}')

        # Copy imagee
        image_copy = image.copy()

        # Template matching
        result = cv2.matchTemplate(image_copy, self.template, cv2.TM_CCORR_NORMED)

        # We want frames where we match at least 80% of the frame to the template
        threshold = 0.8
        loc = np.where(result >= threshold)

        for pt in zip(*loc[::1]):
            print(f'Found one in {pt}')

if __name__ == '__main__':
    Main()