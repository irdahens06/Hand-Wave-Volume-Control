# import cv2
# import mediapipe as mp
# import math
# import numpy as np
# from ctypes import cast,POINTER
# from comtypes import CLSCTX_ALL
# from pycaw.pycaw import AudioUtilities , IAudioEndpointVolume

# #API declaration 
# mp_drwaing = mp.solutions.drawing_utils
# mp_drwaing_styles = mp.solutions.drawing_styles
# mp_hands = mp.solutions.hands

# #volume control library

# devices = AudioUtilities.GetSpeakers()
# interface = devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
# volume = cast(interface,POINTER(IAudioEndpointVolume))
# volRange=volume.GetVolumeRange()
# minVol,maxVol,volBar,volPer = volRange[0],volRange[1],400,0

# #webcam

# wcam,hcam=640,480
# cam = cv2.VideoCapture(0)
# cam.set(3,wcam)
# cam.set(4,hcam)

# #Mediapipe Hand landmark model

# with mp_hands.hands(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
#     while cam.isOpened():
#         success , image = cam.read()
#         image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#         results = hands.process(image)
#         image = cv2.qvtColor(image,cv2.COLOR_RGB2BGR)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drwaing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS,mp_drwaing_styles.get_default_hand_landmarks_styles.get_default_hand_connections_style())

#         #find position of hands
#         lmlist = []
#         if results.multi_hands_landmarks:
#             myHand=results.multi_hand_landmarks[0]
#             for id,lm in enumerate(myHand.landmark):
#                 h,w,c= image.shape
#                 cx,cy = int(lm.x*w),int(lm.y*h)
#                 lmlist.append([id,cx,cy])
#         #thumb and index finger position detection 

#         if len(lmlist)!=0:
#             x1,y1 = lmlist[4][1],lmlist[4][2]
#             x2,y2 = lmlist[8][1],lmlist[8][2]
#             #marking thumb and index finger
#             cv2.circle(image,(x1,y1),15,(255,255,255))
#             cv2.circle(image,(x2,y2),15,(255,255,255))
#             cv2.line(image,(x1,y1),(x2,y2),(0,255,0),3)
#             length= math.hypot(x2-x1,y2-y1)
#             if length<50:
#                 cv2.line(image,(x1,y1),(x2,y2),(0,0,255))
#             vol= np.interp(length,[50,220],minVol,maxVol)
#             volume.SetMasterVolumeLevel(vol,None)
#             volBar = np.interp(length,[50,220],[400,100])
#             volPer = np.interp(length,length[50,220],[0,100])

#             #volume bar
#             cv2.rectangle(image,(50,150),(85,400),(0,0,0),3)
#             cv2.rectangle(image,(50,int(volBar)),(85,400),(0,0,0),cv2.FILLED)
#             cv2.putText(image,f'{int(volPer)}%',(40,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),3)
#         cv2.imshow('handDetector',image)
#         if cv2.waitkey(1) & 0xFF ==ord('q'):
#             break
# cam.release()


import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Setup for drawing and hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Volume control setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()  # e.g. (-65.25, 0.0)
minVol, maxVol = volRange[0], volRange[1]
volBar = 400
volPer = 0

# Webcam settings
wcam, hcam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wcam)
cam.set(4, hcam)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cam.isOpened():
        success, image = cam.read()
        if not success:
            break

        # Flip / optionally mirror (if you want mirror effect)
        # image = cv2.flip(image, 1)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Convert back to BGR for OpenCV display
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks if any
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

        # Build landmark list
        lmlist = []
        if results.multi_hand_landmarks:
            hand0 = results.multi_hand_landmarks[0]
            h, w, c = image.shape
            for id, lm in enumerate(hand0.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])

        # If landmarks detected, do volume logic
        if len(lmlist) != 0:
            # Thumb tip is id 4, index tip is id 8
            x1, y1 = lmlist[4][1], lmlist[4][2]
            x2, y2 = lmlist[8][1], lmlist[8][2]

            # Draw circles / line
            cv2.circle(image, (x1, y1), 15, (255, 255, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), 15, (255, 255, 255), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            length = math.hypot(x2 - x1, y2 - y1)
            # Optionally highlight when fingers are close
            if length < 50:
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Map length to volume range
            vol = np.interp(length, [50, 220], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)

            # Map to UI scale
            volBar = np.interp(length, [50, 220], [400, 100])
            volPer = np.interp(length, [50, 220], [0, 100])

            # Draw volume bar
            cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
            cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
            cv2.putText(image, f'{int(volPer)}%', (40, 450),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

        cv2.imshow('handDetector', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
     