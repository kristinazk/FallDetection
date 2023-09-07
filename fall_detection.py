import numpy as np
import torch
# from matplotlib import pyplot as plt
# import cv2

# from .. import DEVICE # Imagine you have a GPU device and write code using it.


class FallDetection:
    def __init__(self):
        self.vertical_v_start = np.array([10, 10])
        self.vertical_v_end = np.array([10, 50])
        self.cache_size = 30  # Number of frames to calculate moving average
        self.i = None
        self.skeleton_cache = None
        # self.is_fall_arr = []  # Needed for visualisation
        self.overall_score = []  # Overall Moving Average results

    def __call__(self, skeleton_cache):
        '''
            This __call__ function takes a cache of skeletons as input, with a shape of (M x 17 x 2), where M represents the number of skeletons.
            The value of M is constant and represents time. For example, if you have a 7 fps stream and M is equal to 7 (M = 7), it means that the cache length is 1 second.
            The number 17 represents the count of points in each skeleton (as shown in skeleton.png), and 2 represents the (x, y) coordinates.

            This function uses the cache to detect falls.

            The function will return:
                - bool: isFall (True or False)
                - float: fallScore
        '''

        self.skeleton_cache = skeleton_cache  # Needed to use in helper function

        cache_arr = np.zeros(self.cache_size)

        # Defining output variables
        is_fall = False
        fall_score = 0

        # true_checkpoint = 0

        for i in range(len(skeleton_cache) - 1):
            self.i = i

            # 1) angle
            fall_score_cur = self.calc_angle_diff(13, 11, 13, 15)

            # 2) angle
            fall_score_cur += self.calc_angle_diff(14, 12, 14, 16)

            # 3) angle
            fall_score_cur += self.calc_angle_diff(12, 11, 12, 14)

            # 4) angle
            fall_score_cur += self.calc_angle_diff(11, 12, 11, 13)

            # 5) angle
            fall_score_cur += self.calc_angle_diff(6, 8, 6, 12)

            # 6) angle
            fall_score_cur += self.calc_angle_diff(5, 7, 5, 11)

            # 7) angle
            fall_score_cur += 2.8 * self.calc_angle_diff(0, 5, 6, calc_vertical_angles=True)

            # 8) angle
            fall_score_cur += 2.8 * self.calc_angle_diff(0, 11, 12, calc_vertical_angles=True)

            fall_score_cur /= 8

            if i < self.cache_size:
                cache_arr[i] = fall_score_cur
            else:
                cache_arr = np.roll(cache_arr, -1)
                cache_arr[-1] = fall_score_cur

            fall_score = np.nanmean(cache_arr).item()
            self.overall_score.append(fall_score)  # Needed to then find and output the highest fall score

            if round(fall_score, 1) >= 5:
                is_fall = True
            #     self.is_fall_arr.append(True)
            #     true_checkpoint = i
            # else:
            #     is_fall = False
                # self.is_fall_arr.append(False)

        # if any(self.is_fall_arr):
        #     self.is_fall_arr[true_checkpoint:] = [True] * (len(skeleton_cache) - 1 - true_checkpoint)

        # fall_score = np.max(self.overall_score)

        fall_score = round(max(self.overall_score), 1)

        return is_fall, fall_score

    # HELPER FUNCTIONS

    def calc_angle_diff(self, x1, x2, x3, x4=None, calc_vertical_angles=False):
        if calc_vertical_angles:
            angle1 = self.calc_angle(self.skeleton_cache[self.i][x1],
                                     np.array(
                                         (self.skeleton_cache[self.i][x2][0] +
                                          self.skeleton_cache[self.i][x3][0]) / 2,
                                         (self.skeleton_cache[self.i][x2][1] +
                                          self.skeleton_cache[self.i][x3][1]) / 2),
                                     self.vertical_v_start,
                                     self.vertical_v_end)

            angle2 = self.calc_angle(self.skeleton_cache[self.i + 1][x1],
                                     np.array((self.skeleton_cache[self.i + 1][x2][0] +
                                               self.skeleton_cache[self.i + 1][x3][0]) / 2,
                                              (self.skeleton_cache[self.i + 1][x2][1] +
                                               self.skeleton_cache[self.i + 1][x3][1]) / 2),
                                     self.vertical_v_start,
                                     self.vertical_v_end)
            return abs(angle2 - angle1)

        angle1 = self.calc_angle(self.skeleton_cache[self.i][x1],
                                 self.skeleton_cache[self.i][x2],
                                 self.skeleton_cache[self.i][x3],
                                 self.skeleton_cache[self.i][x4])

        angle2 = self.calc_angle(self.skeleton_cache[self.i + 1][x1],
                                 self.skeleton_cache[self.i + 1][x2],
                                 self.skeleton_cache[self.i + 1][x3],
                                 self.skeleton_cache[self.i + 1][x4])

        return abs(angle2 - angle1)

    @staticmethod
    def calc_angle(v1, v2, v3, v4):
        vec_1 = v2 - v1
        vec_2 = v4 - v3

        return np.arccos(np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))) * 180 / np.pi


data1 = np.load('data/skeleton_1.npy')
data2 = np.load('data/skeleton_2.npy')
data3 = np.load('data/skeleton_3.npy')

fd = FallDetection()

# Video Visualisation ##################

# All the necessary elements to run the code below have been commented out for the sake of simplicity. ######
# The produces videos with fall scores can be found in the 'Created Videos' directory.

# def start_cap(vid_path):
#     cap = cv2.VideoCapture(vid_path)
#     return cap
#
#
# def create_writer(cap, output_video_path=None):
#     if not cap.isOpened():
#         print("Error: Could not open video file.")
#         exit()
#
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#
#
#     fourcc = cv2.VideoWriter.fourcc(*'XVID')
#     writer = cv2.VideoWriter(output_video_path,
#                              apiPreference=0,
#                              fourcc=fourcc,
#                              fps=int(fps),
#                              frameSize=(int(width), int(height)),
#                              isColor=True
#                              )
#
#     return writer
#
#
# def video_writer(video_path, output_video_path):
#     cap = start_cap(video_path)
#     writer = create_writer(cap, output_video_path)
#
#     fall_score = fd.overall_score
#     is_fall_arr = fd.is_fall_arr
#
#     for i in range(len(fall_score)):
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         image = cv2.putText(frame,
#                             f'''Is Fall: {str(is_fall_arr[i])}  Fall Score: {str(round(fall_score[i], 4))}''',
#                             (50, 50),
#                             cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                             1,
#                             (255, 255, 255),
#                             2,
#                             cv2.LINE_4)
#
#         writer.write(image)
#
#
# video_writer('data/video_1.mp4', 'Created Videos/video_1_edited.mp4')
