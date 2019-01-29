def kp_mask(img):    
    import cv2
    import dlib
    import matplotlib.pyplot as plt
    detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor('Data/shape_predictor_68_face_landmarks.dat')
    fig = plt.figure(figsize=(20,20))
    new_mask = np.zeros(img.shape, np.uint8)
    faces = detector(img,1)
    area = np.zeros([30, 2], np.uint32)
    if (len(faces) > 0):
        for k,d in enumerate(faces):
            shape = landmark_predictor(img,d)
            show_img = img.copy()
#             for i in range(68):
#                 cv2.circle(show_img, (shape.part(i).x, shape.part(i).y),2,(0,255,0), -1, 8)
#                 plt.subplot(121), plt.imshow(show_img)
            for i in range(17):
                area[i] = np.array([shape.part(i).x, shape.part(i).y])
            area[17] = 1/2 * (np.array([shape.part(16).x, shape.part(16).y]) + np.array([shape.part(26).x, shape.part(26).y]))
            n = 18
            for i in range(5):
                area[n] = np.array([shape.part(26-i).x, shape.part(26-i).y])
                n += 1
            area[n] = 1/2 * (np.array([shape.part(21).x, shape.part(21).y]) + np.array([shape.part(22).x, shape.part(22).y]))
            n += 1
            for i in range(5):
                area[n] = np.array([shape.part(21-i).x, shape.part(21-i).y])
                n += 1
            area[n] = 1/2 * (np.array([shape.part(0).x, shape.part(0).y]) + np.array([shape.part(17).x, shape.part(17).y]))
    cv2.fillPoly(new_mask, [np.int32(area)], (255, 255, 255))
#     plt.subplot(122), plt.imshow(new_mask)
    return new_mask
