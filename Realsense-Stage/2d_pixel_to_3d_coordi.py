# -*- coding: utf-8 -*-
import pyrealsense2 as rs
import numpy as np
import cv2
import json 
import datetime
#--------------------mediapipe area-------------------------------------------------------
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

#----------------------mediapipe end-------------------------------------------------------------------
#--------------------drawing_utils code start -----------
import math
from typing import List, Mapping, Optional, Tuple, Union
#--------------------drawing_utils code end -----------
#testing 
import time

#testing end

pipeline = rs.pipeline()    
config = rs.config()   
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)      
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)     
pipe_profile = pipeline.start(config)       

align_to = rs.stream.color     
align = rs.align(align_to)     


def get_aligned_images():
    
    frames = pipeline.wait_for_frames()     
    aligned_frames = align.process(frames)   
    aligned_depth_frame = aligned_frames.get_depth_frame()     
    aligned_color_frame = aligned_frames.get_color_frame()    

    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics    
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics    


    img_color = np.asanyarray(aligned_color_frame.get_data())   
    img_depth = np.asanyarray(aligned_depth_frame.get_data())      
    return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame


def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    if depth_pixel is None:
        return 0,[0,0,0]
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = aligned_depth_frame.get_distance(x, y)      

    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    # print ('camera_coordinate: ',camera_coordinate)
    return dis, camera_coordinate

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.5
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def get_skeleton_pixel(joints,width, height,skeleton_type):
    skeleton_list=[]
    if joints is not None and skeleton_type==0 and joints.pose_landmarks:
        for idx, landmark in enumerate(joints.pose_landmarks.landmark):

            landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                            width, height)
            #if landmark_px is not None:
            skeleton_list.append([str(idx),landmark_px])

    if joints is not None and skeleton_type==1 and joints.multi_hand_landmarks:
        for idx, landmark in enumerate(joints.multi_hand_landmarks):
            hand_type=joints.multi_handedness[idx].classification[0].label

            for i in range(21):
                landmark_px = _normalized_to_pixel_coordinates(landmark.landmark[i].x, landmark.landmark[i].y,
                                                            width, height)
                #if landmark_px is not None:
                skeleton_list.append([str(hand_type),str(i),landmark_px])


    return skeleton_list

def convert_to_json(pose_coordinate_3d,hand_left_keypoints_3d,hand_right_keypoints_3d,for_department="ME"):
    if for_department=="ME":
        output={
        "timestamp": str(datetime.datetime.now()),
            'pose_keypoints_3d':pose_coordinate_3d, 
            'hand_left_keypoints_3d':hand_right_keypoints_3d, # Handedness is determined assuming the input image is mirrored in mediapipe
            'hand_right_keypoints_3d':hand_left_keypoints_3d  # Handedness is determined assuming the input image is mirrored in mediapipe
        }

    if for_department=="IM":
        pose_coordinate_3d=list(np.transpose([pose_coordinate_3d]).flatten())
        hand_left_keypoints_3d=list(np.transpose([hand_left_keypoints_3d]).flatten())
        hand_right_keypoints_3d=list(np.transpose([hand_right_keypoints_3d]).flatten())
        
        output={
            'pose_keypoints_3d':pose_coordinate_3d, 
            'hand_left_keypoints_3d':hand_right_keypoints_3d, # Handedness is determined assuming the input image is mirrored in mediapipe
            'hand_right_keypoints_3d':hand_left_keypoints_3d  # Handedness is determined assuming the input image is mirrored in mediapipe
        }

    output=json.dumps(output)
    return output

def write_file(json_):
    f = open("skeleton_coordinate_list.txt", "a")
    f.write(json_)
    f.close()



if __name__=="__main__":
    while True:

        pose_skeleton=None
        
        hands_skeleton=None
        t1= time.time()
        color_intrin, depth_intrin, color_image, img_depth, aligned_depth_frame = get_aligned_images()  
#---------------------------------mediapipe start------------------------------------------------
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            color_image.flags.writeable = False
            results = pose.process(color_image)
            pose_skeleton=results
            # Draw the pose annotation on the image.
            color_image.flags.writeable = True
            mp_drawing.draw_landmarks(
                color_image,
                results.pose_landmarks,
                #None,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            with mp_hands.Hands(
                max_num_hands=10,
                model_complexity=0,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3) as hands:
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
                color_image.flags.writeable = False
                results = hands.process(color_image)
                hands_skeleton=results
            # Draw the hand annotations on the image.
                color_image.flags.writeable = True
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            color_image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                #rgb base
                #im_skeleton = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                #cv2.imshow("skeleton", im_skeleton)
#---------------------------------mediapipe end-------------------------------------------------

        #depth_pixel = [320, 240]        center of image
        pose_skeleton_list = get_skeleton_pixel(pose_skeleton,640,480,0)
        hands_skeleton_list = get_skeleton_pixel(hands_skeleton,640,480,1)
        output_list = pose_skeleton_list + hands_skeleton_list
        pose_coordinate_3d=[]
        hand_left_keypoints_3d=[]
        hand_right_keypoints_3d=[]
        for idx,pixel in pose_skeleton_list:
            
            dis, camera_coordinate = get_3d_camera_coordinate(pixel, aligned_depth_frame, depth_intrin)
            if idx=="11" or idx=="13" or idx=="15" or idx == "0":
                print ("index: " + idx + ' pose_coordinate(m): hight(+0.07):weight(+0.14):depth(-0.19)' , camera_coordinate)

            #convert to json use
            #pose_coordinate_3d.append([camera_coordinate])

            #check that every transformative pixel is equal to the original one
            #cv2.circle(color_image, pixel, 8, [255,0,255], thickness=-1)
            #cv2.putText(color_image,"Dis:"+str(dis)+" m", (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[0,0,255])
            #cv2.putText(color_image,"X:"+str(camera_coordinate[0])+" m", (80,80), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,0])
            #cv2.putText(color_image,"Y:"+str(camera_coordinate[1])+" m", (80,120), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,0])
            #cv2.putText(color_image,"Z:"+str(camera_coordinate[2])+" m", (80,160), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,0])
        for hand_type,idx,pixel in hands_skeleton_list:

            dis, camera_coordinate = get_3d_camera_coordinate(pixel, aligned_depth_frame, depth_intrin)
            if hand_type =="Right":
                if idx=="0" or idx=="4" or idx=="12" or idx=="20":
                    print ("hand_type(mirrored): "+hand_type+" index: " + idx + ' hand_coordinate(m): hight:weight:depth' , camera_coordinate) # Handedness is determined assuming the input image is mirrored in mediapipe
            
            #convert to json use
            #if(hand_type =="Left"):
            #    hand_left_keypoints_3d.append([camera_coordinate])
            #if(hand_type =="Right"):
            #    hand_right_keypoints_3d.append([camera_coordinate])

            #check that every transformative pixel is equal to the original one
            #cv2.circle(color_image, pixel, 8, [255,0,255], thickness=-1)
            #cv2.putText(color_image,"Dis:"+str(dis)+" m", (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[0,0,255])
            #cv2.putText(color_image,"X:"+str(camera_coordinate[0])+" m", (80,80), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,0])
            #cv2.putText(color_image,"Y:"+str(camera_coordinate[1])+" m", (80,120), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,0])
            #cv2.putText(color_image,"Z:"+str(camera_coordinate[2])+" m", (80,160), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,0])
        t2 = time.time()

        print("time",t2-t1)
#--------------------------------------------------write to files start------------------
        #json_=convert_to_json(pose_coordinate_3d,hand_left_keypoints_3d,hand_right_keypoints_3d,"IM")
        #check = json.loads(json_)
        #print(check["hand_left_keypoints_3d"])
        #write_file(json_)
#--------------------------------------------------write to files end------------------

        cv2.imshow('RealSence',color_image)
        key = cv2.waitKey(1)