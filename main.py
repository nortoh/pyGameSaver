import ffmpeg
import logging
import sys, getopt
import cv2
import time
import numpy as np

class Main(object):

    def __init__(self):
        print("template_matcher (nortoh) v1.0")
        self.video = 'movie.mp4'
        self.template_image_file = 'template2.png'
        self.template = cv2.imread(self.template_image_file)

    def execute(self):
        cam = cv2.VideoCapture(self.video)

        start_time = time.perf_counter()

        # FPS and frames
        fps = cam.get(cv2.CAP_PROP_FPS)
        frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        # duration
        duration = frame_count / fps
        duration_minutes = int(duration / 60)
        duration_seconds = duration % 60
        print(f'FPS: {fps} frame_count: {frame_count}')
        
        # Stick to a small section of frames for debugging
        current_frame = 0
        
        while(current_frame < frame_count):
            ret, image = cam.read()

            if ret:
                # We have a good frame
                check_time = time.perf_counter()
                # Check frame
                # print(f'Checking frame: {current_frame}')
                self.check_frame(current_frame, image)
                
                if (current_frame % 100 == 1):
                    # duration
                    finished_time = current_frame / fps
                    finished_minutes = int(finished_time / 60)
                    finished_seconds = finished_time % 60

                    # timer
                    end_time = time.perf_counter()
                    run_time = end_time - start_time
                    match_time = end_time - check_time
                    print(f'[({current_frame}/{frame_count}) Runtime: {run_time:0.4f}] ({str(round(finished_minutes, 2))}:{str(round(finished_seconds, 2))}/{str(round(duration_minutes, 2))}:{str(round(duration_seconds, 2))})')

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
        results = zip(*loc[::1])

        # print(results, sep=',')
        # if len(loc) != 0:
        #     print(f"Found something on frame {current_frame}")
        
        
        for pt in zip(*loc[::1]):
            print(f'Found one in frame {current_frame} - {pt}')

if __name__ == '__main__':
    Main().execute()