# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

"""
OpenGL Pointcloud viewer with http://pyglet.org
Usage:
------
Mouse:
    Drag with left button to rotate around pivot (thick small axes),
    with right button to translate and the wheel to zoom.
Keyboard:
    [p]     Pause
    [r]     Reset View
    [d]     Cycle through decimation values
    [z]     Toggle point scaling
    [x]     Toggle point distance attenuation
    [c]     Toggle color source
    [l]     Toggle lighting
    [f]     Toggle depth post-processing
    [s]     Save PNG (./out.png)
    [e]     Export points to ply (./out.ply)
    [q/ESC] Quit
Notes:
------
Using deprecated OpenGL (FFP lighting, matrix stack...) however, draw calls 
are kept low with pyglet.graphics.* which uses glDrawArrays internally.
Normals calculation is done with numpy on CPU which is rather slow, should really
be done with shaders but was omitted for several reasons - brevity, for lowering
dependencies (pyglet doesn't ship with shader support & recommends pyshaders)
and for reference.
"""

import math
import ctypes
import pyglet
import pyglet.gl as gl
import numpy as np
import pyrealsense2 as rs
#--------------------mediapipe area-------------------------------------------------------
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

#----------------------mediapipe end-------------------------------------------------------------------
import pandas as pd


# pyglet 宣告使用
window = pyglet.window.Window(
    config=gl.Config(
        double_buffer=True,
        samples=8  # MSAA
    ),
    resizable=True, vsync=True)
keys = pyglet.window.key.KeyStateHandler()
window.push_handlers(keys)


# https://stackoverflow.com/a/6802723
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

# 宣告操作的基本變數
class AppState:

    def __init__(self, *args, **kwargs):
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, 1], np.float32)
        self.distance = 2
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 0
        self.scale = True
        self.attenuation = False
        self.color = True
        self.lighting = False
        self.postprocessing = False

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, 1

    @property
    def rotation(self):
        Rx = rotation_matrix((1, 0, 0), math.radians(-self.pitch))
        Ry = rotation_matrix((0, 1, 0), math.radians(-self.yaw))
        return np.dot(Ry, Rx).astype(np.float32)

# 轉換顏色
def convert_fmt(fmt):
    """轉換顏色 : rs.format to pyglet format string"""
    return {
        rs.format.rgb8: 'RGB',
        rs.format.bgr8: 'BGR',
        rs.format.rgba8: 'RGBA',
        rs.format.bgra8: 'BGRA',
        rs.format.y8: 'L',
    }[fmt]

# 滑鼠操作
def handle_mouse_btns(x, y, button, modifiers):
    '''
    滑鼠操作
    '''
    state.mouse_btns[0] ^= (button & pyglet.window.mouse.LEFT)
    state.mouse_btns[1] ^= (button & pyglet.window.mouse.RIGHT)
    state.mouse_btns[2] ^= (button & pyglet.window.mouse.MIDDLE)

# 鍵盤操作
def on_key_press(symbol, modifiers):
    '''
    鍵盤操作
    '''
    if symbol == pyglet.window.key.R:
        state.reset()

    if symbol == pyglet.window.key.P:
        state.paused ^= True

    if symbol == pyglet.window.key.D:
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

    if symbol == pyglet.window.key.C:
        state.color ^= True

    if symbol == pyglet.window.key.Z:
        state.scale ^= True

    if symbol == pyglet.window.key.X:
        state.attenuation ^= True

    if symbol == pyglet.window.key.L:
        state.lighting ^= True

    if symbol == pyglet.window.key.F:
        state.postprocessing ^= True

    if symbol == pyglet.window.key.S:
        pyglet.image.get_buffer_manager().get_color_buffer().save('out.png')

    if symbol == pyglet.window.key.Q:
        window.close()

# 繪製 3D 軸
def axes(size=1, width=1):
    """繪製 3D 軸:draw 3d axes"""
    gl.glLineWidth(width)
    pyglet.graphics.draw(6, gl.GL_LINES,
                         ('v3f', (0, 0, 0, size, 0, 0,
                                  0, 0, 0, 0, size, 0,
                                  0, 0, 0, 0, 0, size)),
                         ('c3f', (1, 0, 0, 1, 0, 0,
                                  0, 1, 0, 0, 1, 0,
                                  0, 0, 1, 0, 0, 1,
                                  ))
                         )

# 錐台
def frustum(intrinsics):
    """錐/影響觀看(Perspective/FOV):draw camera's frustum
    
    https://zh.wikipedia.org/zh-tw/%E9%94%A5%E5%8F%B0
    """
    w, h = intrinsics.width, intrinsics.height
    batch = pyglet.graphics.Batch()

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            batch.add(2, gl.GL_LINES, None, ('v3f', [0, 0, 0] + p))
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

       
        batch.add(2, gl.GL_LINES, None, ('v3f', top_left + top_right))
        batch.add(2, gl.GL_LINES, None, ('v3f', top_right + bottom_right))
        batch.add(2, gl.GL_LINES, None, ('v3f', bottom_right + bottom_left))
        batch.add(2, gl.GL_LINES, None, ('v3f', bottom_left + top_left))

    batch.draw()

def draw_skeleton(intrinsics,depth_colormap,joints):
    """draw skeleton"""
    w, h = intrinsics.width, intrinsics.height
    batch = pyglet.graphics.Batch()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    if(joints.pose_landmarks):
        for idx, landmark in enumerate(joints.pose_landmarks.landmark):
            depth = depth_colormap[landmark.y,landmark.x].astype(float)
            depth = depth * depth_scale
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [landmark.x, landmark.y], depth)
            batch.add(2, gl.GL_POINTS, None, ('v3f',[0, 0, 0] + p))
            return p


    batch.draw()

def find_skeleton_verts(points,joints):
    """draw skeleton"""
    verts=np.asarray(points.get_vertices())
    print(verts)
    skelton_verts=[]
    info = verts.tolist()
    df = pd.DataFrame(info,columns=["x","y","z"])
    if(joints.pose_landmarks):
        print(len(joints.pose_landmarks.landmark))
        for idx, landmark in enumerate(joints.pose_landmarks.landmark):
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
            
            # Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
            depth_sensor =pipeline_profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
        
            # Map depth to color
            depth_pixel = [240, 320]   # Random pixel
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_scale)
            print(depth_point)
            #skelton_verts=df[(round(df["x"],6)==round(landmark.x,6)) & (round(df["y"],6)==round(landmark.y,6))]
            test=df[(round(df["x"],3)==round(landmark.x,3)) & (round(df["y"],3)==round(landmark.y,3))]
            if not test.empty:
                skelton_verts.append(test)

        #dict={"xy":[]}
        #dict["xy"].append( verts[0])
        #if(joints.pose_landmarks):
        
        #    for i in verts[0]:
        #        for j in verts[1]:
        #            if [i,j] in joints.pose_landmarks.landmark:
        #                skelton_verts.append(vert)
       #
       

#
                    
        return skelton_verts
    else :
        return verts

# 繪製 3D 面網格線
def grid(size=1, n=10, width=1):
    """繪製 3D 面網格線:draw a grid on xz plane"""
    gl.glLineWidth(width)
    s = size / float(n)
    s2 = 0.5 * size
    batch = pyglet.graphics.Batch()

    for i in range(0, n + 1):
        x = -s2 + i * s
        batch.add(2, gl.GL_LINES, None, ('v3f', (x, 0, -s2, x, 0, s2)))
    for i in range(0, n + 1):
        z = -s2 + i * s
        batch.add(2, gl.GL_LINES, None, ('v3f', (-s2, 0, z, s2, 0, z)))

    batch.draw()

# OPENGL 繪製流程
@window.event
def on_draw():
    window.clear()

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_LINE_SMOOTH)

    width, height = window.get_size()
    gl.glViewport(0, 0, width, height)

    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.gluPerspective(60, width / float(height), 0.01, 20)

    gl.glMatrixMode(gl.GL_TEXTURE)
    gl.glLoadIdentity()
    # texcoords are [0..1] and relative to top-left pixel corner, add 0.5 to center
    gl.glTranslatef(0.5 / image_data.width, 0.5 / image_data.height, 0)
    image_texture = image_data.get_texture()
    # texture size may be increased by pyglet to a power of 2
    tw, th = image_texture.owner.width, image_texture.owner.height
    gl.glScalef(image_data.width / float(tw),
                image_data.height / float(th), 1)

    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()

    gl.gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)

    gl.glTranslatef(0, 0, state.distance)
    gl.glRotated(state.pitch, 1, 0, 0)
    gl.glRotated(state.yaw, 0, 1, 0)

    if any(state.mouse_btns):
        axes(0.1, 4)

    gl.glTranslatef(0, 0, -state.distance)
    gl.glTranslatef(*state.translation)

    gl.glColor3f(0.5, 0.5, 0.5)
    gl.glPushMatrix()
    gl.glTranslatef(0, 0.5, 0.5)
    grid()
    gl.glPopMatrix()

    # 顆粒大小
    psz = max(window.get_size()) / float(max(w, h)) if state.scale else 1
    #psz = 3
    #print('psz:',psz)
    gl.glPointSize(psz)
    distance = (0, 0, 1) if state.attenuation else (1, 0, 0)
    gl.glPointParameterfv(gl.GL_POINT_DISTANCE_ATTENUATION,
                          (gl.GLfloat * 3)(*distance))

    if state.lighting:
        ldir = [0.5, 0.5, 0.5]  # world-space lighting
        ldir = np.dot(state.rotation, (0, 0, 1))  # MeshLab style lighting
        ldir = list(ldir) + [0]  # w=0, directional light
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (gl.GLfloat * 4)(*ldir))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE,
                     (gl.GLfloat * 3)(1.0, 1.0, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT,
                     (gl.GLfloat * 3)(0.75, 0.75, 0.75))
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_NORMALIZE)
        gl.glEnable(gl.GL_LIGHTING)

    gl.glColor3f(1, 1, 1)
    texture = image_data.get_texture()
    gl.glEnable(texture.target)
    gl.glBindTexture(texture.target, texture.id)
    gl.glTexParameteri(
        gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

    # comment this to get round points with MSAA on
    gl.glEnable(gl.GL_POINT_SPRITE)

    if not state.scale and not state.attenuation:
        gl.glDisable(gl.GL_MULTISAMPLE)  # for true 1px points with MSAA on
    # 繪製點雲 PointCloud
    vertex_list.draw(gl.GL_POINTS)
    gl.glDisable(texture.target)
    if not state.scale and not state.attenuation:
        gl.glEnable(gl.GL_MULTISAMPLE)

    gl.glDisable(gl.GL_LIGHTING)

    gl.glColor3f(0.25, 0.25, 0.25)
    frustum(depth_intrinsics)
    axes()

    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(0, width, 0, height, -1, 1)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    gl.glMatrixMode(gl.GL_TEXTURE)
    gl.glLoadIdentity()
    gl.glDisable(gl.GL_DEPTH_TEST)

    # 最後繪畫步驟
    fps_display.draw()

# 滑鼠操作事件
@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    w, h = map(float, window.get_size())

    if buttons & pyglet.window.mouse.LEFT:
        state.yaw -= dx * 0.5
        state.pitch -= dy * 0.5

    if buttons & pyglet.window.mouse.RIGHT:
        dp = np.array((dx / w, -dy / h, 0), np.float32)
        state.translation += np.dot(state.rotation, dp)

    if buttons & pyglet.window.mouse.MIDDLE:
        dz = dy * 0.01
        state.translation -= (0, 0, dz)
        state.distance -= dz

# 滑鼠操作事件
@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    dz = scroll_y * 0.1
    state.translation -= (0, 0, dz)
    state.distance -= dz


def run(dt):
    '''
    主程式運作區
    '''
    # 初始設定
    global w, h
    window.set_caption("RealSense (%dx%d) %dFPS (%.2fms) %s" %
                       (w, h, 0 if dt == 0 else 1.0 / dt, dt * 1000,
                        "PAUSED" if state.paused else ""))

    if state.paused:
        return

    success, frames = pipeline.try_wait_for_frames(timeout_ms=0)
    if not success:
        return
    
    # 讀取 RealSense 來源
    depth_frame = frames.get_depth_frame().as_video_frame()
    other_frame = frames.first(other_stream).as_video_frame()

    depth_frame = decimate.process(depth_frame)

    if state.postprocessing:
        for f in filters:
            depth_frame = f.process(depth_frame)

    # Grab new intrinsics (may be changed by decimation)
    depth_intrinsics = rs.video_stream_profile(
        depth_frame.profile).get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    color_image = np.asanyarray(other_frame.get_data())
    im_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    #cv2.imshow("rgb", im_rgb)
    cv2.imwrite('./png/raw.jpg', im_rgb)
    
#-------------------mediapipe start---------
    # 人體語意分割
    BG_COLOR =(0,0,0) # 黑色
    BG_COLOR = (192, 192, 192) # gray
    with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=1) as selfie_segmentation:
        bg_image = None
#
        color_image.flags.writeable = False
        results = selfie_segmentation.process(color_image)
#
        color_image.flags.writeable = True
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('./png/raw_convColor.jpg', color_image)
        # Draw selfie segmentation on the background image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack(
        (results.segmentation_mask,) * 3, axis=-1) > 0.1
        # The background can be customized.
        #   a) Load an image (with the same width and height of the input image) to
        #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
        #   b) Blur the input image by applying image filtering, e.g.,
        #      bg_image = cv2.GaussianBlur(image,(55,55),0)
        if bg_image is None:
            bg_image = np.zeros(color_image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
        color_image = np.where(condition, color_image, bg_image)
        #color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('./png/segmentation.jpg', color_image)
    # 人體骨架
        skeleton=None
    #    with mp_pose.Pose(
    #        min_detection_confidence=0.5,
    #        min_tracking_confidence=0.5) as pose:
 #
    #        # To improve performance, optionally mark the image as not writeable to
    #        # pass by reference.
    #        color_image.flags.writeable = False
    #        results = pose.process(color_image)
    #        skeleton=results
    #        # Draw the pose annotation on the image.
    #        color_image.flags.writeable = True
    #        mp_drawing.draw_landmarks(
    #            color_image,
    #            results.pose_landmarks,
    #            #None,
    #            mp_pose.POSE_CONNECTIONS,
    #            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    #        # Flip the image horizontally for a selfie-view display.
    #        #cv2.imwrite('./png/pose.jpg', color_image)
    ##手部判斷
    #    with mp_hands.Hands(
    #        model_complexity=0,
    #        min_detection_confidence=0.5,
    #        min_tracking_confidence=0.5) as hands:
##
    #    # To improve performance, optionally mark the image as not writeable to
    #    # pass by reference.
    #        color_image.flags.writeable = False
    #        #cv2.imshow("skeleton", color_image)
    #        results = hands.process(color_image)
##
    #    # Draw the hand annotations on the image.
    #    color_image.flags.writeable = True
    #    if results.multi_hand_landmarks:
    #        for hand_landmarks in results.multi_hand_landmarks:
    #            mp_drawing.draw_landmarks(
    #                color_image,
    #                hand_landmarks,
    #                mp_hands.HAND_CONNECTIONS,
    #                mp_drawing_styles.get_default_hand_landmarks_style(),
    #                mp_drawing_styles.get_default_hand_connections_style())
    #    cv2.imwrite('./png/pose_hand.jpg', color_image)
        #rgb base
        #im_skeleton = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        #cv2.imshow("skeleton", color_image)
    

    #-------------------mediapipe end----------

    colorized_depth = colorizer.colorize(depth_frame)
    depth_colormap = np.asanyarray(colorized_depth.get_data())
    #cv2.imshow("depth", depth_colormap)
    #if(skeleton is not None):
        #draw_skeleton(depth_intrinsics,depth_colormap,skeleton)
    
    # 影像來源為:color_source-------------------------------------------------------------
    if state.color:
        mapped_frame, color_source = other_frame, color_image
    else:
        mapped_frame, color_source = colorized_depth, depth_colormap

    points = pc.calculate(depth_frame)
    pc.map_to(mapped_frame)

    # 處理顏色跟圖像大小轉換 : handle color source or size change
    fmt = convert_fmt(mapped_frame.profile.format())
    #fmt = 'BGR'
    global image_data

    if (image_data.format, image_data.pitch) != (fmt, color_source.strides[0]):
        if state.color:
            global color_w, color_h
            image_w, image_h = color_w, color_h
        else:
            image_w, image_h = w, h

        empty = (gl.GLubyte * (image_w * image_h * 3))()
        image_data = pyglet.image.ImageData(image_w, image_h, fmt, empty)

    # 來源色彩資料至 OPENGL copy image data to pyglet
    image_data.set_data(fmt, color_source.strides[0], color_source.ctypes.data)
    #verts=find_skeleton_verts(points,skeleton)
    verts = np.asarray(points.get_vertices(2)).reshape(h, w, 3)
    #verts = np.array_split(verts,2)[0]
    #verts_new = np.zeros((int(h/2), int(w), 3))
    #verts = np.concatenate((verts, verts_new), axis=0)
    #verts = np.full((h, w, 3),10)
    #verts[int(h/2):int(h/2+10)][int(w/2):int(w/2+10)][0] = 10
    #verts[h/2:h/2+10][w/2:w/2+10][1] = 20
    #verts[int(h/2)][int(w/2)][0] = 30
    ## 這邊需要做點轉換過濾器, 非人點Point 過濾-------------------------------------------------------------
    texcoords = np.asarray(points.get_texture_coordinates(2))
    #print("verts",verts)
    if len(vertex_list.vertices) != verts.size:
        vertex_list.resize(verts.size // 3)
        # need to reassign after resizing
        vertex_list.vertices = verts.ravel()
        vertex_list.tex_coords = texcoords.ravel()

    # copy our data to pre-allocated buffers, this is faster than assigning...
    # pyglet will take care of uploading to GPU
    def copy(dst, src):
        """copy numpy array to pyglet array"""
        # timeit was mostly inconclusive, favoring slice assignment for safety
        np.array(dst, copy=False)[:] = src.ravel()
        # ctypes.memmove(dst, src.ctypes.data, src.nbytes)

    copy(vertex_list.vertices, verts)
    copy(vertex_list.tex_coords, texcoords)

    if state.lighting:
        # compute normals
        dy, dx = np.gradient(verts, axis=(0, 1))
        n = np.cross(dx, dy)

        # can use this, np.linalg.norm or similar to normalize, but OpenGL can do this for us, see GL_NORMALIZE above
        # norm = np.sqrt((n*n).sum(axis=2, keepdims=True))
        # np.divide(n, norm, out=n, where=norm != 0)

        # import cv2
        # n = cv2.bilateralFilter(n, 5, 1, 1)

        copy(vertex_list.normals, n)



    if keys[pyglet.window.key.E]:
        # PLY是一種電腦檔案格式，全名為多邊形檔案（Polygon File Format） => 非點雲, 為表面積組成立體檔案
        # https://zh.m.wikipedia.org/zh-tw/PLY
        points.export_to_ply('./out.ply', mapped_frame) 





if __name__ == '__main__':
    
    state = AppState()

    # Configure streams
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    other_stream, other_format = rs.stream.color, rs.format.rgb8
    config.enable_stream(other_stream, other_format, 30)

    # Start streaming
    pipeline.start(config)
    profile = pipeline.get_active_profile()

    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    # Processing blocks
    pc = rs.pointcloud()
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
    colorizer = rs.colorizer()
    # Filter 素人範例:https://blog.csdn.net/Dontla/article/details/103574458
    filters = [rs.disparity_transform(),
            rs.spatial_filter(),
            rs.temporal_filter(),
            rs.disparity_transform(False)]






    # Create a VertexList to hold pointcloud data
    # Will pre-allocates memory according to the attributes below
    vertex_list = pyglet.graphics.vertex_list(
        w * h, 'v3f/stream', 't2f/stream', 'n3f/stream')
    # Create and allocate memory for our color data
    other_profile = rs.video_stream_profile(profile.get_stream(other_stream))

    image_w, image_h = w, h
    color_intrinsics = other_profile.get_intrinsics()
    color_w, color_h = color_intrinsics.width, color_intrinsics.height

    if state.color:
        image_w, image_h = color_w, color_h

    image_data = pyglet.image.ImageData(image_w, image_h, convert_fmt(
    other_profile.format()), (gl.GLubyte * (image_w * image_h * 3))())

    #cv2.imshow('image_data',image_data)


    if (pyglet.version <  '1.4' ):
        # pyglet.clock.ClockDisplay has be removed in 1.4
        fps_display = pyglet.clock.ClockDisplay()
    else:
        fps_display = pyglet.window.FPSDisplay(window)


    window.on_mouse_press = window.on_mouse_release = handle_mouse_btns
    window.push_handlers(on_key_press)


    pyglet.clock.schedule(run)

    try:
        pyglet.app.run()
    finally:
        pipeline.stop()