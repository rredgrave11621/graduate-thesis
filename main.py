from fastapi import FastAPI, File, UploadFile
from tempfile import NamedTemporaryFile
from fastapi.concurrency import run_in_threadpool
import aiofiles
import traceback
import os
import cv2
# import torch
from einops import rearrange
import mediapipe as mp
import onnxruntime
import numpy as np
from dictionary import label

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
app = FastAPI()
ort_session = onnxruntime.InferenceSession("slr_aagcn.onnx")
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name 

@app.post("/video_async")
async def video_async(file: UploadFile = File(...)):
    try:
        async with aiofiles.tempfile.NamedTemporaryFile("wb", delete=False) as temp:
            try:
                contents = await file.read()
                await temp.write(contents)
            except Exception:
                return {"message": "There was an error uploading the file"}
            finally:
                await file.close()
        
        # res = await run_in_threadpool(process_video, temp.name)  # Pass temp.name to VideoCapture()
        res = await process_video(temp.name)
    except Exception:
        traceback.print_exc()
        return {"message": "There was an error processing the file"}
    finally:
        os.remove(temp.name)

    return res

async def process_video(path):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    keypoints = await extract_keypoints2(path)
    # model.eval()
    # with torch.no_grad():
        # y_hat = model.to(device)(keypoints)
    # label_pred, confidence = torch.argmax(y_hat), torch.max(y_hat)
    # print(label_pred)
    ort_inputs = {input_name: keypoints}
    ort_outs = ort_session.run([output_name], ort_inputs)
    output_word, confidence = np.argmax(ort_outs[0]).item(), max(ort_outs[0][0]).item()
    word = label[output_word]
    return {"predict" : word, "confidence": confidence}

async def extract_keypoints2(video_path):
    # Input : video_path
    # Output : N, C, T, V, 1
    video = cv2.VideoCapture(video_path)
    count = 0
    pose_start = 0
    pose_end = 0
    # Get all keypoints
    list_keypoints = []
    while True:
            # keypoint_dict : Dict[str, List[float]] = {}
            ret, frame = video.read()
            

            if not ret:
                break

            count += 1

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = image.shape[0], image.shape[1]

            if count == 1:
                results = holistic.process(image)
                if results.pose_landmarks:
                    # Crop pose only 
                    print("Right shoulder", results.pose_landmarks.landmark[12].x)
                    print("Left shoulder", results.pose_landmarks.landmark[11].x)
                    width_man = abs(results.pose_landmarks.landmark[12].x - results.pose_landmarks.landmark[11].x)
                    scale = 1.5
                    pose_start = int(width * (results.pose_landmarks.landmark[12].x - scale * width_man)) 
                    pose_end = int(width * (results.pose_landmarks.landmark[11].x + scale * width_man)) 
                    print(width_man)
                    print(pose_start)
                    print(pose_end)
                else:
                    pose_start = 0
                    pose_end = width

            image = image[0:height, pose_start:pose_end]
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Smoothing image
            image = cv2.GaussianBlur(image, (5, 5), 0)
            # Check if right hand landmarks are found
            if results.right_hand_landmarks:
                # Iterate over each right hand landmark
                for landmark in results.right_hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    list_keypoints.append(x)
                    list_keypoints.append(y)
            else:
                for i in range(21):
                    list_keypoints.append(0)
                    list_keypoints.append(0)

            # Check if left hand landmarks are found
            if results.left_hand_landmarks:
                # Iterate over each right hand landmark
                for landmark in results.left_hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    list_keypoints.append(x)
                    list_keypoints.append(y)
            else:
                for i in range(21):
                    list_keypoints.append(0)
                    list_keypoints.append(0)
            
            if results.pose_landmarks:
                for idx in [30, 12, 2, 20]: 
                    landmark = results.pose_landmarks.landmark[idx]
                    x = landmark.x
                    y = landmark.y
                    list_keypoints.append(x)
                    list_keypoints.append(y)
            else:
                for idx in range(4):
                    list_keypoints.append(0)
                    list_keypoints.append(0)

    # Reshape 
    print(count)
    # keypoints = torch.tensor(list_keypoints)
    keypoints = np.array(list_keypoints)
    # .astype(np.float32)
    # Split frame
    keypoints = rearrange(keypoints, '(a b) -> a b', a = count)
    print(keypoints.shape)
    # Split node
    keypoints = rearrange(keypoints, 'a (b c) -> a b c', b = 46)
    print(keypoints.shape)
    if count < 80:
        # target = torch.zeros(80, 46, 2)
        target = np.zeros((80, 46, 2))
        target[:count, :, :] = keypoints
    else:
        # target = torch.zeros(80, 46, 2)
        target = np.zeros((80, 46, 2))
        target = keypoints[:80, :, :]
    print(target.shape)
    # target = target.unsqueeze(axis = -1)
    target = np.expand_dims(target, axis=-1)
    target = rearrange(target, ' t v c 1 -> c t v 1')
    print(target.shape)
    # target = target.unsqueeze(axis = 0)
    target = np.expand_dims(target, axis=0).astype(np.float32)
    print(target.shape)
    return target
