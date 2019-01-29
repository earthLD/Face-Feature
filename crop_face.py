def crop_face(img):
    import os
    import dlib
    from skimage import io
    import numpy as np
    detector_path ='Data/mmod_human_face_detector.dat'
    face_detector = dlib.cnn_face_detection_model_v1(detector_path)
    detected_faces = face_detector(img)
    if len(detected_faces)<1:
        return None
    d = detected_faces[0].rect
    left = d.left(); right = d.right(); top = d.top(); bottom = d.bottom()
    high = bottom-top
    k = right-left
    x, y, _ = img.shape
    x1 = int(max(top-high/2, 0))
    x2 = int(min(bottom+high/2, x))
    y1 = int(max(left-k/2, 0))
    y2 = int(min(right+k/2, y))
    img_crop = img[x1:x2, y1:y2, :]
    return img_crop
