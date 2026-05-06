import cv2
import os
import dlib
#import FaceAlign as FA
import numpy as np
import lightgbm as lgb
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import f1_score, roc_curve, auc
from scipy.spatial import distance as dist
import pandas as pd
import tensorflow as tf
import mediapipe as mp

from skimage.feature import hog
#from skimage import data, exposure
from skimage.transform import resize
import sys

models_dir = os.path.join(os.path.dirname(__file__), "models")
predictor_path = os.path.join(models_dir, "shape_predictor_5_face_landmarks.dat")

_sp = None


def set_models_dir(path):
    global models_dir, predictor_path, _sp
    models_dir = path
    predictor_path = os.path.join(models_dir, "shape_predictor_5_face_landmarks.dat")
    _sp = None


def get_shape_predictor():
    global _sp
    if _sp is None:
        _sp = dlib.shape_predictor(predictor_path)
    return _sp


def _resolve_models_dir():
    if os.path.exists(predictor_path):
        return
    env_path = os.environ.get("PYAFAR_MODELS_DIR")
    if env_path and os.path.exists(os.path.join(env_path, "shape_predictor_5_face_landmarks.dat")):
        set_models_dir(env_path)
        return
    try:
        import PyAFAR_GUI
        pkg_models = os.path.join(os.path.dirname(PyAFAR_GUI.__file__), "models")
        if os.path.exists(os.path.join(pkg_models, "shape_predictor_5_face_landmarks.dat")):
            set_models_dir(pkg_models)
    except Exception:
        pass


def head_posr(face_landmarks, img_w, img_h):
    face_2d = []
    face_3d = []

    for idx, lm in enumerate(face_landmarks.landmark):
        
        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
            if idx == 1:
                nose_2d = (lm.x * img_w, lm.y * img_h)
                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                

            x, y = int(lm.x * img_w), int(lm.y * img_h)

            # Get the 2D Coordinates
            face_2d.append([x, y])

            # Get the 3D Coordinates
            face_3d.append([x, y, lm.z])       
        
    # Convert it to the NumPy array
    face_2d = np.array(face_2d, dtype=np.float64)

    # Convert it to the NumPy array
    face_3d = np.array(face_3d, dtype=np.float64)

    # The camera matrix
    focal_length = 1 * img_w

    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                [0, focal_length, img_w / 2],
                                [0, 0, 1]])

    # The distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)


    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)
        
    #xc = rotationMatrixToEulerAngles(rmat)

    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    # Get the y rotation degree
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    return x,y,z


def dlibalign(landmarks, img):
    faces = dlib.full_object_detections()
    landmax = np.amax(np.array(landmarks), axis=0)
    landmin = np.amin(np.array(landmarks),axis=0)
    detection = dlib.rectangle(left=landmin[0], top=landmin[1], right=landmax[0], bottom=landmax[1])
    faces.append(get_shape_predictor()(img, detection))
    return dlib.get_face_chip(img, faces[0], size=250)
    
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	#print(eye[1])
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

lipsUpperInner = [78,  81, 311,  308]
lipsLowerInner = [78,  178, 402, 308]

lips = [78,  81, 311,  308,  402, 178]

LEFT_EYE_b = [362,385,387,263,373,380]
RIGHT_EYE_b = [33,160,158,133,153,144]



mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def set_max_num_faces(max_num_faces):
    global face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=int(max_num_faces),
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def crop_img(image):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    #print(image.shape)
    
    results = face_mesh.process(image)
    
    image.flags.writeable = True
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    img_h, img_w, img_c = image.shape
    imgs = []
    all_lands = []
    
    if results.multi_face_landmarks:
                
        
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []


            for idx, lm in enumerate(face_landmarks.landmark):
                landmarks.append([int(lm.x*img_w),int(lm.y*img_h)])
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                
            #new_img = conveximg(image, landmarks, img_w, img_h)
            new_img = dlibalign(landmarks, image)
            #print(new_img)
            all_lands.append(face_landmarks)
            imgs.append(new_img)
        return imgs, all_lands
        
    else:
        return None


def infant_afar_multi(filename, AUs, GPU, max_frames, fill_value=0):
    _resolve_models_dir()
    print(filename)
    print(os.path.exists(filename))
    #print(PID)
    cap = cv2.VideoCapture(filename)
    #cap.release()
    blak_fm = np.zeros((224, 224, 3), np.uint8)
    print("Length of video")
    print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fffnum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    predictions = {}
    people = {}
    flist = []
    face_idx_list = []
    if GPU:
       device = '0'
    else:
       device = "-1"

    person_ids = []
    predictions['int'] = {}
    predictions['occ'] = {}
    predictions['Frame'] = []
    predictions['Person_ID'] = []
    predictions['Pitch'] = []
    predictions['Yaw'] = []
    predictions['Roll'] = []
    predictions['EAR'] = []
    predictions['MAR'] = []
    skiped_frames = []
    for i in range(478):
        predictions["x_"+str(i)] = []
        predictions["y_"+str(i)] = []
        predictions["z_"+str(i)] = []
    
    for i in AUs:
       predictions['occ'][i] = []
    
    frames_no = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    all_frames = []
    valid_mask = []
    cnt = -1
    while cap.isOpened():
        ret, frame = cap.read()
        cnt +=1
        #print(cnt)
       
        #print(ret)
        if not ret:
            break
        try:
            img_w, img_h, img_c = frame.shape
            result = crop_img(frame)
            if result is None:
                raise ValueError("No face detected")
            ft_list, lands = result
            if ft_list is None or lands is None or len(ft_list) == 0 or len(lands) == 0:
                raise ValueError("No face detected")

            for ln in range(len(lands)):
                ft = ft_list[ln]
                # Extract HoG features on the gray image of size 112 x 112
                aligned_img = cv2.resize(ft, (224,224))
                gray_image = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
                image_resized = resize(gray_image, (gray_image.shape[0] // 2, gray_image.shape[1] // 2), anti_aliasing=True)
                fd = hog(image_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
                all_frames.append(fd.tolist())
                valid_mask.append(True)
                flist.append(cnt)
                face_idx_list.append(ln)

                lpland = []
                eyeL = []
                eyeR = []
                landmarks = []
                x, y, z = head_posr(lands[ln], frame.shape[0], frame.shape[1])
                for idx, lm in enumerate(lands[ln].landmark):
                    predictions["x_"+str(idx)].append(lm.x)
                    predictions["y_"+str(idx)].append(lm.y)
                    predictions["z_"+str(idx)].append(lm.z)

                    landmarks.append([int(lm.x*img_w),int(lm.y*img_h)])

                    if idx in lips:
                        lpland.append((lm.x*img_w, lm.y*img_h))
                    if idx in LEFT_EYE_b:
                        eyeL.append((lm.x*img_w, lm.y*img_h))
                    if idx in RIGHT_EYE_b:
                        eyeR.append((lm.x*img_w, lm.y*img_h))

                ear = eye_aspect_ratio(lpland)
                predictions["MAR"].append(ear)
                Left_ear = eye_aspect_ratio(eyeL)
                Right_ear = eye_aspect_ratio(eyeR)
                predictions["EAR"].append((Left_ear+Right_ear)/2.0)
                predictions["Pitch"].append(x)
                predictions["Yaw"].append(y)
                predictions["Roll"].append(z)
        except Exception as e:
            print(e)
            skiped_frames.append(cnt)
            flist.append(cnt)
            face_idx_list.append(-1)
            ft = blak_fm.copy()
            aligned_img = cv2.resize(ft, (224,224))
            gray_image = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
            image_resized = resize(gray_image, (gray_image.shape[0] // 2, gray_image.shape[1] // 2), anti_aliasing=True)
            fd = hog(image_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            all_frames.append(fd.tolist())
            valid_mask.append(False)
            predictions["MAR"].append(fill_value)
            predictions["EAR"].append(fill_value)
            predictions["Pitch"].append(fill_value)
            predictions["Yaw"].append(fill_value)
            predictions["Roll"].append(fill_value)
            for idx in range(468):
                predictions["x_"+str(idx)].append(fill_value)
                predictions["y_"+str(idx)].append(fill_value)
                predictions["z_"+str(idx)].append(fill_value)
             
        if len(all_frames) >= max_frames:
            for k in AUs:
                model_path = os.path.join(models_dir ,"Infant",  k+".txt")
                model = lgb.Booster(model_file=model_path)
                preds = model.predict(np.array(all_frames)).tolist()
                for i, p in enumerate(preds):
                    predictions["occ"][k].append(p if valid_mask[i] else fill_value)
            del  all_frames
            all_frames = []
            del valid_mask
            valid_mask = []
            
    if len(all_frames) > 0:
        for k in AUs: 
            model_path = os.path.join(models_dir ,"Infant",  k+".txt")
            model = lgb.Booster(model_file=model_path)
            preds = model.predict(np.array(all_frames)).tolist()
            for i, p in enumerate(preds):
                predictions["occ"][k].append(p if valid_mask[i] else fill_value)
    
    # CHANGED HERE
    # to_pd = {}
    # to_pd["Frame"] = [gh for gh in range(fffnum)]
    # to_pd["Pitch"] = predictions["Pitch"]
    # to_pd["Yaw"] = predictions["Yaw"]
    # to_pd["Roll"] = predictions["Roll"]
    # to_pd["Eye Aspect Ratio"] = predictions["EAR"]
    # to_pd["Mouth Aspect Ratio"] = predictions["MAR"]
    # for k in AUs:
    #     to_pd["Occ_"+k] = predictions["occ"][k]
    #     for j in skiped_frames: 
    #         to_pd["Occ_"+k][j] = -1
    # for k in range(468):
    #     to_pd["x_"+str(k)] = predictions["x_"+str(k)]
    #     to_pd["y_"+str(k)] = predictions["y_"+str(k)]
    #     to_pd["z_"+str(k)] = predictions["z_"+str(k)]

    to_pd = {}
    to_pd["Frame"] = flist[:]
    to_pd["Face_Index"] = face_idx_list[:]
    n = len(flist)

    def _pad(a, n, fill=fill_value):
        # ensure same length; trims or pads as needed
        return a[:n] if len(a) >= n else a + [fill] * (n - len(a))

    to_pd["Pitch"] = _pad(predictions["Pitch"], n)
    to_pd["Yaw"]   = _pad(predictions["Yaw"],   n)
    to_pd["Roll"]  = _pad(predictions["Roll"],  n)
    to_pd["Eye Aspect Ratio"]   = _pad(predictions["EAR"], n)
    to_pd["Mouth Aspect Ratio"] = _pad(predictions["MAR"], n)

    for k in AUs:
        to_pd["Occ_"+k] = _pad(predictions["occ"][k], n)

    for idx in range(468):
        to_pd[f"x_{idx}"] = _pad(predictions[f"x_{idx}"], n)
        to_pd[f"y_{idx}"] = _pad(predictions[f"y_{idx}"], n)
        to_pd[f"z_{idx}"] = _pad(predictions[f"z_{idx}"], n)

        
    

    return to_pd
    #'''


if __name__ == "__main__":
    #(filename, AUs, GPU, batchsize, max_frames,AU_Ints)
    fg = infant_afar_multi("videos/FN035_FF_B_AV_473.avi", ["au_6"], True, 1000)
    for k in fg:
       print(k)
       print(len(fg[k]))
    #fg =pd.DataFrame.from_dict(fg)
    fg =pd.DataFrame.from_dict(fg)
    print(fg.head())
    
    
                
                
