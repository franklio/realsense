# -*- coding: utf-8 -*-
import pyrealsense2 as rs
import numpy as np
import cv2
import json
import csv
from datetime import datetime
# --------------------drawing_utils code start -----------
import math
from typing import List, Mapping, Optional, Tuple, Union
# --------------------drawing_utils code end -----------
import time


def get_aligned_images():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    aligned_color_frame = aligned_frames.get_color_frame()

    # 獲取深度參數(用於後續轉相機坐標)
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    # 獲取相機內參
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics
    # RGB圖
    img_color = np.asanyarray(aligned_color_frame.get_data())
    # 深度圖
    img_depth = np.asanyarray(aligned_depth_frame.get_data())
    return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame


def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    if depth_pixel is None:
        return 0, [0, 0, 0]
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = aligned_depth_frame.get_distance(x, y)

    camera_coordinate = rs.rs2_deproject_pixel_to_point(
        depth_intrin, depth_pixel, dis)
    # print ('camera_coordinate: ',camera_coordinate)
    return dis, camera_coordinate


if __name__ == '__main__':
    try:
        w, h = 640, 480
        # initialize (prepare realsense)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
        pipe_profile = pipeline.start(config)
        align_to = rs.stream.color
        align = rs.align(align_to)

        # w, h = int(round(w/2)), int(round(h/2))
        while True:
            # coordinates for validation
            pixels = [[int(w/2), int(h/2)],
                      [int(w/4), int(h/2)],
                      [int(w/4*3), int(h/2)],
                      [int(w/2), int(h/4)],
                      [int(w/2), int(h/4*3)]]

            # get aligned image
            color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = get_aligned_images()

            # obtain depth of defined coordinates
            dis, camera_coordinate = [], []
            for pixel in pixels:
                dis_, camera_coordinate_ = get_3d_camera_coordinate(
                    pixel, aligned_depth_frame, depth_intrin)
                dis.append(round(dis_, 6))
                camera_coordinate.append(camera_coordinate_)
            print(pixels,':',dis)

            # 三維座標, 驗證用不到
            # coordinates = (' '.join([str(round(i, 3))
            #                for i in camera_coordinate]))

            # draw
            # (left up(y,x), right down(y,x))
            img_color = cv2.line(img_color, (0, h), (w, 0), (0, 0, 255), 1)
            img_color = cv2.line(img_color, (0, 0), (w, h), (0, 0, 255), 1)

            # points (for val)
            '''
            A: int(w/2), int(h/2)   -> middle
            B: int(w/4), int(h/2)   -> left
            C: int(w/4*3), int(h/2) -> right
            D: int(w/2), int(h/4)   -> up
            E: int(w/2), int(h/4*3) -> down
            '''
            img_color = cv2.circle(
                img_color, (int(w/2), int(h/2)), 5, (0, 0, 255), -1)
            img_color = cv2.circle(
                img_color, (int(w/4), int(h/2)), 5, (0, 0, 255), -1)
            img_color = cv2.circle(
                img_color, (int(w/4*3), int(h/2)), 5, (0, 0, 255), -1)
            img_color = cv2.circle(
                img_color, (int(w/2), int(h/4)), 5, (0, 0, 255), -1)
            img_color = cv2.circle(
                img_color, (int(w/2), int(h/4*3)), 5, (0, 0, 255), -1)

            # text (results)
            img_color = cv2.rectangle(
                img_color, (0, 0), (350, 25), (0, 0, 0), -1)
            img_color = cv2.putText(img_color, 'distance: ' + str(dis), (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1, cv2.LINE_AA)
            # img_color = cv2.putText(img_color, 'coordinate: ' + str(coordinates),
            #                         (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('Depth in RealSense', img_color)
            key = cv2.waitKey(1)

            if key == ord('s'):  # press 's' to save image & write to csv
                time_ = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
                cv2.imwrite(str(time_) + '.jpg', img_color)

                with open('./output000.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([str(time_), dis])
            if key == ord('q') or key == 27:
                pipeline.stop()
                break

        cv2.destroyAllWindows()

    except RuntimeError:
        print('There are no more frames left!')

    finally:
        pass
