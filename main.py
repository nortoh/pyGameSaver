import ffmpeg
import logging
import sys, getopt
import cv2
import time
import queue
import numpy as np
import multiprocessing as mp
from multiprocessing import Process

from collections import deque

class Main(object):

    def __init__(self):
        print("template_matcher (nortoh) v1.0")

        # Data structures
        self.image_frames = deque()
        self.found_frames = deque()

        # Settings
        self.video = 'movie.mp4'
        self.template_image_file = 'template2.png'
        self.template = cv2.imread(self.template_image_file)

    def start_processes(self):
        self.load_video_process = Process(target=self.load_video)
        self.template_match_process = Process(target=self.perform_vide_check)
        self.load_video_process.start()
        self.template_match_process.start()

    def load_video(self):
        cam = cv2.VideoCapture(self.video)

        start_time = time.perf_counter()

        # FPS and frames ggh 
        self.fps = cam.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        # Width and height

        self.frame_width = int(cam.get(3))
        self.frame_height = int(cam.get(4)) 

        # duration
        duration = self.frame_count / self.fps
        duration_minutes = int(duration / 60)
        duration_seconds = duration % 60
        print(f'FPS: {self.fps} frame_count: {self.frame_count}')
        
        # Stick to a small section of frames for debugging
        current_frame = 0
        max_historic_count = 1000

        while(current_frame < self.frame_count):
            ret, image = cam.read()

            # Handle frame
            if ret:
                # We have a good frame
                check_time = time.perf_counter()

                self.image_frames.append((current_frame, image))

                if (current_frame % 100 == 1):
                    # duration
                    finished_time = current_frame / self.fps
                    finished_minutes = int(finished_time / 60)
                    finished_seconds = finished_time % 60

                    # timer
                    end_time = time.perf_counter()
                    run_time = end_time - start_time
                    match_time = end_time - check_time
                    print(f'[({current_frame}/{self.frame_count}) Runtime: {run_time:0.4f}] ({str(round(finished_minutes, 2))}:{str(round(finished_seconds, 2))}/{str(round(duration_minutes, 2))}:{str(round(duration_seconds, 2))})')
                
                current_frame += 1
            else:
                # We have a bad frame
                break
            
            # We need to start pushing out frames we don't need
            if len(self.image_frames) > max_historic_count:
                self.image_frames.pop()

        # Clean up
        cam.release()
        cv2.destroyAllWindows()
        
        self.clip_and_save()

        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Total runtime: {run_time:0.4f} seconds")

    def perform_check(self):
        while len(self.image_frames) != 0:
            current_frame, image = self.image_frames.get()

            self.check_frame(current_frame, image)


    def check_frame(self, current_frame, image):
        # print(f'current_frame: {str(current_frame)} frame: {image}')

        # Copy imagee
        image_copy = image.copy()

        # Save all frame images to a buffer
        self.image_frames.append(image)

        # Template matching
        result = cv2.matchTemplate(image_copy, self.template, cv2.TM_CCORR_NORMED)

        # We want frames where we match at least 80% of the frame to the template
        threshold = 0.8
        loc = np.where(result >= threshold)
        results = zip(*loc[::1])

        # Throw the match frames into a buffer
        for pt in zip(*loc[::1]):
            print(f'Found one in frame {current_frame} - {pt}')
            self.found_frames.append(current_frame)

    def clip_and_save(self):
        output_video = cv2.VideoWriter('found.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (self.frame_width, self.frame_height))
        left_padding = self.fps * 180 # 5 mins
        
        for found_index, found_frame in enumerate(self.found_frames):

            # Loop through all the video frames
            for index in range(found_frame - left_padding, found_frame):
                output_video.write(self.image_frames[index])
                print(f'[Index: {index}]')

        output_video.release()

if __name__ == '__main__':
    Main().start_processes()