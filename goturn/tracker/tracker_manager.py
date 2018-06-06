# Date: Wednesday 07 June 2017 11:28:11 AM IST
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: tracker manager

import cv2 
import os
import time
import subprocess

opencv_version = cv2.__version__.split('.')[0]

class tracker_manager:


    """Docstring for tracker_manager. """

    def __init__(self, videos, regressor, tracker, logger):
        """This is

        :videos: list of video frames and annotations
        :regressor: regressor object
        :tracker: tracker object
        :logger: logger object
        :returns: list of video sub directories
        """

        self.videos = videos
        self.regressor = regressor
        self.tracker = tracker
        self.logger = logger


    def trackAll(self, start_video_num, pause_val):
        """Track the objects in the video
        """
        output_path = "/datasets/home/64/564/nthui/output/"

        videos = self.videos
        objRegressor = self.regressor
        objTracker = self.tracker
        logger = self.logger

        video_keys = list(videos.keys())
        for video_number in range(start_video_num, len(videos)):
            video_frames = videos[video_keys[video_number]][0]
            annot_frames = videos[video_keys[video_number]][1]

            num_frames = min(len(video_frames), len(annot_frames))

            # Get the first frame of this video with the intial ground-truth bounding box
            frame_0 = video_frames[0]
            bbox_0 = annot_frames[0]
            sMatImage = cv2.imread(frame_0)
            objTracker.init(sMatImage, bbox_0, objRegressor)
            video_output_dir = os.path.join(output_path, video_keys[video_number])
            if not os.path.exists(video_output_dir):
                os.mkdir(video_output_dir)
            total_time = 0
            for i in range(1, num_frames):
                frame = video_frames[i]
                sMatImage = cv2.imread(frame)
                sMatImageDraw = sMatImage.copy()
                bbox = annot_frames[i]
                
                if opencv_version == '2':
                    cv2.rectangle(sMatImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), (255, 255, 255), 2)
                else:
                    sMatImageDraw = cv2.rectangle(sMatImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), (255, 255, 255), 2)
                start = time.time()
                bbox = objTracker.track(sMatImage, objRegressor)
                end = time.time()
                total_time += end - start

                if opencv_version == '2':
                    cv2.rectangle(sMatImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), (255, 0, 0), 2)
                else:
                    sMatImageDraw = cv2.rectangle(sMatImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), (255, 0, 0), 2)

                # cv2.imshow('Results', sMatImageDraw)
                img_path = os.path.join(video_output_dir, "%04d.jpg" % (i))
                self.logger.debug("Writing %s" % (img_path))
                cv2.imwrite(img_path, sMatImageDraw)
                # cv2.waitKey(10)
            results_data_path = os.path.join(output_path, video_keys[video_number], "result.txt")
            results_file = open(results_data_path, "w")
            results_file.write(str(total_time / (num_frames - 1)))
            results_file.close()
            command = "ffmpeg -framerate 30 -y -i %s %s" % (os.path.join(video_output_dir, "%04d.jpg"), os.path.join(video_output_dir, "%s.mp4" % video_keys[video_number]))
            subprocess.call(command, shell = True)

