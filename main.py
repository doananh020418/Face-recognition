import cv2
import face_recognition
import os
import glob
import numpy as np
import time

def load_img(path):
    img_list = []
    name_list = []
    for im in path:
        img = face_recognition.load_image_file(im)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(0, 0), None, 0.25, 0.25)
        img_list.append(img)
        name_list.append(os.path.splitext(im)[0])
    # img = cv2.resize(img,(int(img.shape[0]/2),int(img.shape[1]/2)))
    return img_list, name_list


def show(img, faceLocation, name):
    if len(faceLocation)!=0:
        cv2.rectangle(img, (faceLocation[3]*4, faceLocation[0]*4), (faceLocation[1]*4, faceLocation[2]*4), (0, 255, 255),
                      1)
        cv2.putText(img, name, (faceLocation[1]*4 + 6, faceLocation[2]*4 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    cv2.imshow('Camera', img)
    cv2.waitKey(1)


def processing(img_list):
    encoded_list = []
    loc_list = []
    for img in img_list:
        faceLocation = face_recognition.face_locations(img)[0]
        encodedImg = face_recognition.face_encodings(img)[0]
        encoded_list.append(np.array(encodedImg))
        loc_list.append(faceLocation)
    return encoded_list, loc_list


def encode(img):
    faceLocation = face_recognition.face_locations(img)
    encodedImg = face_recognition.face_encodings(img,faceLocation)
    return encodedImg, faceLocation


def main():
    cap = cv2.VideoCapture(0)
    path = glob.glob('*.jpg')
    #print(path)
    img_list, name_list = load_img(path)
    print(name_list[0])
    knownImg, knownLoc = processing(img_list)
    # show(img_list[0],knownLoc[0],name_list[0])
    name = ''
    imgLoc = []

    ptime = 0
    while True:
        succ, img = cap.read()
        if succ:
            img = cv2.flip(img, 1)
            imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
            imgEncode, imgLoc = encode(imgs)
            #print(imgEncode)
            ctime = time.time()
            fps = 1/(ctime - ptime)
            ptime = ctime
            cv2.putText(img,f'FPS: {int(fps)}',(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
            for imgE,imgL in zip(imgEncode,imgLoc):
                match = face_recognition.compare_faces(knownImg, np.array(imgE))
                faceDis = face_recognition.face_distance(knownImg, np.array(imgEncode))
                matchIdx = np.argmin(faceDis)
                if match[matchIdx]:
                    name = name_list[matchIdx].upper()
                    show(img, imgL, name)


if __name__ == '__main__':
    main()
