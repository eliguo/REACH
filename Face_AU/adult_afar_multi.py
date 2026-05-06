import numpy as np
import os
import sys 
import json 
from tensorflow.keras.models import load_model
import dlib 
import cv2
from scipy.spatial import distance as dist
import pandas as pd
import tensorflow as tf
import mediapipe as mp
import skimage
from tensorflow.keras import layers, models

from tensorflow.keras.models import model_from_json



models_dir = os.path.join(os.path.dirname(__file__), "models")
predictor_path = os.path.join(models_dir, "shape_predictor_5_face_landmarks.dat")

'''
facenet_path = os.path.join(models_dir, 'facenet_architecture.json')

facenet_model_path = os.path.join(models_dir, 'facenet_wt.h5')

with open(facenet_path, 'r') as json_file:
    loaded_model_json = json_file.read()

facenet_model = model_from_json(loaded_model_json)

facenet_model.load_weights(facenet_model_path)
'''

#facenet_model = FaceNet()

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

def whiten_image(img):
  img = img/255.0
  axis = (0, 1, 2)
  size = img.size
  mean = np.mean(img, axis=axis, keepdims=True)
  std = np.std(img, axis=axis, keepdims=True)
  std_adj = np.maximum(std, 1.0/np.sqrt(size))
  whitened_image = (img-mean)/std
  return whitened_image

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


def encode_face(img):
  #only the face part
  face_only = img
  #cv2.imshow("test222", face_only)
  # resize to 160x160 (without stretching)
  face_only = skimage.transform.resize(face_only,(160, 160), mode='reflect')
  # normalize
  face_only = whiten_image(face_only)
  # find encoding
  encoding = facenet_model.predict(face_only)
  return encoding


def find_closest_face(encoding, names_faces):
  distances = []
  for name, face in names_faces.items():
    dist = np.linalg.norm(encoding-face)
    distances.append(dist)
  min_distance = np.min(distances)
  min_distance_name = list(names_faces)[np.argmin(distances)]
  return min_distance, min_distance_name


def tracker(img, people, landmarks):
  numpers = len(people)
  landmax = np.amax(np.array(landmarks), axis=0)
  landmin = np.amin(np.array(landmarks),axis=0)
  encoding = encode_face(img[landmin[1]:landmax[1], landmin[0]:landmax[0], :])
  if not (numpers ==0):
    dist, name = find_closest_face(encoding, people)

  else:
    dist = 1000
  if (dist <10):
    return name, people
  else:
    #print(dist)
    people[str(numpers+1)] = encoding
    return str(numpers+1), people


def conveximg(img, landmarks, width, height):
    convexhull = cv2.convexHull(np.array(landmarks))
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, convexhull, 255)
    face_extracted = cv2.bitwise_and(img, img, mask=mask)
    return face_extracted


def dlibalign(landmarks, img):
    faces = dlib.full_object_detections()
    landmax = np.amax(np.array(landmarks), axis=0)
    landmin = np.amin(np.array(landmarks),axis=0)
    detection = dlib.rectangle(left=landmin[0], top=landmin[1], right=landmax[0], bottom=landmax[1])
    faces.append(get_shape_predictor()(img, detection))
    return dlib.get_face_chip(img, faces[0], size=250)
    

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
    # --- STRONG TYPE GUARD: only accept np.ndarray frames ---
    if image is None:
        return None  # or raise, your choice
    image = np.asarray(image)
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
                
            new_img = conveximg(image, landmarks, img_w, img_h)
            new_img = dlibalign(landmarks, new_img)
            #print(new_img)
            all_lands.append(face_landmarks)
            imgs.append(new_img)
            
        return imgs, all_lands
        
    else:
        return None
    
lipsUpperInner = [78,  81, 311,  308]
lipsLowerInner = [78,  178, 402, 308]

lips = [78,  81, 311,  308,  402, 178]

LEFT_EYE_b = [362,385,387,263,373,380]
RIGHT_EYE_b = [33,160,158,133,153,144]

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


def predict_AU_Adult_int(device, k, int_frames, batch_size):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    model_path = os.path.join(models_dir, "adult", "int", k+".h5")
    model = load_model(model_path)
    num_samples = len(int_frames)
    num_batches = int(np.ceil(len(int_frames) / batch_size))

    predictions = []

    for i in range(num_batches): 
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_input = int_frames[start_idx:end_idx]
        batch_pred = model.predict(batch_input)
        batch_pred = np.array(batch_pred)
        batch_pred = np.argmax(batch_pred, axis=1)
        predictions.append(batch_pred)
    
    predictions = np.concatenate(predictions, axis=0)
    print(predictions.shape)
    predictions =predictions.tolist()

    return predictions
      


def predict_AU_Adult_occ(device, k, occ_frames, batch_size):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    model_path = os.path.join(models_dir ,"adult",'occ',  k+".h5")
    model = load_model(model_path)
    num_samples = len(occ_frames)
    num_batches = int(np.ceil(len(occ_frames) / batch_size))

    predictions = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_input = occ_frames[start_idx:end_idx]
        batch_pred = model.predict(batch_input)
        predictions.append(batch_pred)

    # Concatenate predictions from all batches into a single array
    predictions = np.concatenate(predictions, axis=0)
    predictions = predictions[:,0].tolist()

    return predictions
    

def adult_afar_multi(filename, AUs, GPU, max_frames, AU_Int, batch_size, PID, fill_value=0):
    _resolve_models_dir()
    print(filename)
    print(os.path.exists(filename))
    print(PID)
    cap = cv2.VideoCapture(filename)
    #cap.release()
    blak_fm = np.zeros((250, 250, 3), np.uint8)
    print("Length of video")
    print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
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
    start_fram = 0


    for i in range(478):
        predictions["x_"+str(i)] = []
        predictions["y_"+str(i)] = []
        predictions["z_"+str(i)] = []
    
    for i in AU_Int:
       predictions['int'][i] = []

    for i in AUs:
       predictions['occ'][i] = []
    
    frames_no = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_fram = 0
    
    num_batches = int(frames_no / max_frames)+1
    skiped_frame = []
    all_frames = []
    valid_mask = []
    cnt = -1
    while cap.isOpened():
        ret, frame = cap.read()
        cnt += 1
        if not ret or frame is None:
            if cnt > 0:
                break
            else:
                continue

        frame = np.asarray(frame)
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
                land = lands[ln]
                all_frames.append(ft)
                valid_mask.append(True)
                flist.append(cnt)
                face_idx_list.append(ln)

                lpland = []
                eyeL = []
                eyeR = []
                landmarks = []
                x, y, z = head_posr(land, frame.shape[0], frame.shape[1])
                for idx, lm in enumerate(land.landmark):
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

                if PID:
                    pname, people = tracker(frame, people, landmarks)
                    person_ids.append(pname)

        except Exception as e:
            print(e)
            skiped_frames.append(cnt)
            all_frames.append(blak_fm.copy())
            valid_mask.append(False)
            flist.append(cnt)
            face_idx_list.append(-1)
            predictions["MAR"].append(fill_value)
            predictions["EAR"].append(fill_value)
            predictions["Pitch"].append(fill_value)
            predictions["Yaw"].append(fill_value)
            predictions["Roll"].append(fill_value)
            for idx in range(468):
                predictions["x_"+str(idx)].append(fill_value)
                predictions["y_"+str(idx)].append(fill_value)
                predictions["z_"+str(idx)].append(fill_value)
            if PID:
                person_ids.append(fill_value)

        if len(all_frames) >= max_frames:
            X = np.stack(all_frames, axis=0)
            batch_mask = valid_mask[:]

            for k in AUs:
                preds = predict_AU_Adult_occ(device, k, X, batch_size)
                for i, p in enumerate(preds):
                    predictions["occ"][k].append(p if batch_mask[i] else fill_value)

            for k in AU_Int:
                preds = predict_AU_Adult_int(device, k, X, batch_size)
                for i, p in enumerate(preds):
                    predictions["int"][k].append(p if batch_mask[i] else fill_value)

            del all_frames
            all_frames = []
            del valid_mask
            valid_mask = []


    if len(all_frames) > 0:
        X = np.stack(all_frames, axis=0)
        batch_mask = valid_mask[:]

        for k in AUs:
            preds = predict_AU_Adult_occ(device, k, X, batch_size)
            for i, p in enumerate(preds):
                predictions["occ"][k].append(p if batch_mask[i] else fill_value)

        for k in AU_Int:
            preds = predict_AU_Adult_int(device, k, X, batch_size)
            for i, p in enumerate(preds):
                predictions["int"][k].append(p if batch_mask[i] else fill_value)
       


    to_pd = {}
    to_pd["Frame"] = flist
    to_pd["Face_Index"] = face_idx_list
    if PID: 
       to_pd["Person_ID"] = person_ids

    to_pd["Pitch"] = predictions["Pitch"]
    to_pd["Yaw"] = predictions["Yaw"]
    to_pd["Roll"] = predictions["Roll"]
    to_pd["Eye Aspect Ratio"] = predictions["EAR"]
    to_pd["Mouth Aspect Ratio"] = predictions["MAR"]
    for k in AUs:
        to_pd["Occ_"+k] = predictions["occ"][k]
        #for j in skiped_frames: 
        #    to_pd["Occ_"+k][j] = -1

    for k in AU_Int:
        to_pd["Int_"+k] = predictions["int"][k]
        #for j in skiped_frames: 
        #    to_pd["Int_"+k][j] = -1

    for k in range(468):
        to_pd["x_"+str(k)] = predictions["x_"+str(k)]
        to_pd["y_"+str(k)] = predictions["y_"+str(k)]
        to_pd["z_"+str(k)] = predictions["z_"+str(k)]

    return to_pd

if __name__ == "__main__":
    import time
    start_time = time.time()
    #adult_occ_aus = [1,2,4,6,7,10,12,14,15,17,23,24]
    #adult_int_aus = [6,10,12,14,17]

    fg = adult_afar_multi(r"D:\Codes\Pitt\2024\Ext PyAFAR\inp\LeftVideoSN003_comp.avi", ["au_1"], True, 1000, ["au_6"], 100, False)
    
    

    '''
    for k in fg:
       print(k)
       print(len(fg[k]))
    '''
    fg =pd.DataFrame.from_dict(fg)
    #print(fg.head())
    end_time = time.time()
    print(end_time - start_time)





    
        
       
