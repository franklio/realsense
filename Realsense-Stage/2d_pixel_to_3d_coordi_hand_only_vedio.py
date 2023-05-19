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
#--------judge skin color--------------
import colorsys
#--------judge skin color--------------

#------------config----------------------
#video_input=r"C:\Users\a5566\OneDrive\Desktop\realsense_record\20230314_151135.bag"
video_input=r"C:\Users\a5566\OneDrive\Desktop\git\realsense\20230314_145135.bag"
input_image_resolusion_weight=1280
input_image_resolusion_height=720

#------------config----------------------
pipeline = rs.pipeline()    
config = rs.config()   
rs.config.enable_device_from_file(config, video_input)

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
other_stream, other_format = rs.stream.color, rs.format.rgb8
config.enable_stream(other_stream, other_format, 30)
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


#-------------use skin color to judge different hand------------------------------------------
def get_color(img,x, y):
    #im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('pixel', x, y)
    b, g, r = img[y, x]
    print("r,g,b", r, g, b)
    return r,g,b

def color_distance(color1, color2):
    #ref https://stackoverflow.com/questions/35113979/calculate-distance-between-colors-in-hsv-space
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    hsv1 = colorsys.rgb_to_hsv(r1/255.0, g1/255.0, b1/255.0)
    hsv2 = colorsys.rgb_to_hsv(r2/255.0, g2/255.0, b2/255.0)
    hue1, sat1, val1 = hsv1
    hue2, sat2, val2 = hsv2
    hue_diff = min(abs(hue1 - hue2), 1.0 - abs(hue1 - hue2))
    hue_diff_radians = hue_diff * math.pi * 2.0
    sat_diff = sat1 - sat2
    val_diff = val1 - val2
    return math.sqrt(hue_diff_radians ** 2 + sat_diff ** 2 + val_diff ** 2)
#-------------use skin color to judge different hand end-----------------------------------------

if __name__=="__main__":

    frame=1
    master_right_hand_pixel,master_right_hand_rgb=[0,0],[255,48,48]
    master_left_hand_pixel,master_left_hand_rgb=[0,0],[255,48,48]
    ex_frame_right=[]
    ex_frame_left=[]
    ex_frame2=[]
    ex_frame3=[]
    ex_frame4=[]
    ex_frame5=[]
    ex_frame6=[]
    ex_frame7=[]
    ex_frame8=[]
    ex_frame9=[]
    ex_frame10=[]
    while True:
        hands_skeleton=None
        
        color_intrin, depth_intrin, color_image, img_depth, aligned_depth_frame = get_aligned_images()  
#---------------------------------mediapipe start------------------------------------------------
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
        hands_skeleton_list = get_skeleton_pixel(hands_skeleton,input_image_resolusion_weight,input_image_resolusion_height,1)
        pose_coordinate_3d=[]
        hand_left_keypoints_3d=[]
        hand_right_keypoints_3d=[]
        left_hand_distance_list=[]
        left_distance=100
        right_distance=100
        lower_bound=100
        upper_bound=100
        is_left_multiple=0
        is_right_multiple=0
        for hand_type,idx,pixel in hands_skeleton_list:     
            dis, camera_coordinate = get_3d_camera_coordinate(pixel, aligned_depth_frame, depth_intrin)
            #print ("frame: "+ str(frame) +" hand_type(mirrored): "+hand_type+" index: " + idx + ' hand_coordinate(m): ' , camera_coordinate) # Handedness is determined assuming the input image is mirrored in mediapipe

            
            if(idx=="0"):
                print ("frame: "+ str(frame) +" hand_type(mirrored): "+hand_type+" index: " + idx + ' hand_coordinate(m): ' , camera_coordinate) # Handedness is determined assuming the input image is mirrored in mediapipe
                r,g,b=get_color(color_image,pixel[0],pixel[1])
                if hand_type== "Left" :
                    is_left_multiple=1
                    ##-------------use skin color to judge different hand-----------------------------------------
                    #distance = color_distance(master_left_hand_rgb, [r,g,b])
                    #if distance<left_distance:
                    #    left_distance=distance
                    #    master_left_hand_rgb,master_left_hand_pixel=[r,g,b],pixel
                    ##-------------use skin color to judge different hand end--------------------------------------

                    #-------------use pixel location to judge different hand-----------------------------------------
                    if(master_left_hand_pixel[0]==0 and master_left_hand_pixel[1]==0):
                        master_left_hand_pixel=pixel
                    elif (master_left_hand_pixel[0]-lower_bound<pixel[0] 
                            and pixel[0]<master_left_hand_pixel[1]+upper_bound 
                            and master_left_hand_pixel[1]-lower_bound<pixel[1] 
                            and pixel[1]<master_left_hand_pixel[1]+upper_bound):
                        master_left_hand_pixel=pixel

                    if(len(ex_frame_left)>=10):
                        ex_frame_left.pop()
                    ex_frame_left.append(master_left_hand_pixel)
                    print("ex_frame_left",ex_frame_left)
                    #-------------use pixel location to judge different hand-----------------------------------------
                    
                if hand_type== "Right" :
                    is_right_multiple=1
                    ##-------------use skin color to judge different hand-----------------------------------------
                    #distance = color_distance(master_right_hand_rgb, [r,g,b])
                    #if distance<right_distance:
                    #    right_distance=distance
                    #    master_right_hand_rgb,master_right_hand_pixel=[r,g,b],pixel
                    ##-------------use skin color to judge different hand end--------------------------------------
                    #-------------use pixel location to judge different hand-----------------------------------------
                    if(master_right_hand_pixel[0]==0 and master_right_hand_pixel[1]==0):
                        master_right_hand_pixel=pixel
                    elif (master_right_hand_pixel[0]-lower_bound<pixel[0] 
                            and pixel[0]<master_right_hand_pixel[1]+upper_bound 
                            and master_right_hand_pixel[1]-lower_bound<pixel[1] 
                            and pixel[1]<master_right_hand_pixel[1]+upper_bound):
                        master_right_hand_pixel=pixel

                    

                    if (len(ex_frame_right)>=10):
                        ex_frame_right.pop()
                    ex_frame_right.append(master_right_hand_pixel)
                    print("ex_frame_right",ex_frame_right)
                    #-------------use pixel location to judge different hand-----------------------------------------

                    
                    

            #convert to json use
            #if(h   and_type =="Left"):
            #    hand_left_keypoints_3d.append([camera_coordinate])
            #if(hand_type =="Right"):
            #    hand_right_keypoints_3d.append([camera_coordinate])

            #check that every transformative pixel is equal to the original one
            #cv2.circle(color_image, pixel, 8, [255,0,255], thickness=-1)
            #cv2.putText(color_image,"Dis:"+str(dis)+" m", (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[0,0,255])
            #cv2.putText(color_image,"X:"+str(camera_coordinate[0])+" m", (80,80), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,0])
            #cv2.putText(color_image,"Y:"+str(camera_coordinate[1])+" m", (80,120), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,0])
            #cv2.putText(color_image,"Z:"+str(camera_coordinate[2])+" m", (80,160), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,0])

        cv2.circle(color_image,master_left_hand_pixel, 15, (255, 0, 0), -1)
        cv2.circle(color_image,master_right_hand_pixel, 15, (255, 0, 0), -1)

        frame+=1

            
#--------------------------------------------------write to files start------------------
        #json_=convert_to_json(pose_coordinate_3d,hand_left_keypoints_3d,hand_right_keypoints_3d,"IM")
        #check = json.loads(json_)
        #print(check["hand_left_keypoints_3d"])
        #write_file(json_)
#--------------------------------------------------write to files end------------------

        cv2.imshow('RealSence',cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        key = cv2.waitKey(1)