import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#Camera feed
webcam = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while webcam.isOpened():
        ret, frame = webcam.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        def calcular_angulo(a, b, c):
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)

            radiante = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angulo = np.abs(radiante*180/np.pi)
            if angulo > 180:
                angulo = angulo - 360
            return angulo


        # extrair landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            angulo = calcular_angulo(shoulder, elbow, wrist)

            cv2.putText(image, str(angulo),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA
                        )

            if angulo > 40:
                msg = "contracao de menos 40"
                cor = (0, 0, 255)
            else:
                msg = "contracao de mais 40"
                cor = (0, 255, 0)

            cv2.putText(image, msg,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, cor, 2, cv2.LINE_AA)
        except: pass



        cv2.imshow("ugabuga", image)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break



    webcam.release()
    cv2.destroyAllWindows()

