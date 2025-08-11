import cv2
import mediapipe as mp
from sklearn.utils import shuffle
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import os

path = 'attention.mp4'
folder_path1 = 'attention'
folder_path2 = 'atease'
folder_path3 = 'test'

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

all_landmarks = []
all_labels = []
test_landmarks = []

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    image_files1 = [f for f in os.listdir(folder_path1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files2 = [f for f in os.listdir(folder_path2) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files1:
        img_path = os.path.join(folder_path1, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print("Could not read image")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        try:
            landmarks = results.pose_landmarks.landmark
            landmark_data = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
            all_landmarks.append(landmark_data)
            all_labels.append(1) 
        except:
            continue

    for img_file in image_files2:
        img_path = os.path.join(folder_path2, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print("Could not read image")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        try:
            landmarks = results.pose_landmarks.landmark
            landmark_data = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
            all_landmarks.append(landmark_data)
            all_labels.append(0) 
        except:
            continue

    image_files3 = [f for f in os.listdir(folder_path3) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in image_files3:
        img_path = os.path.join(folder_path3, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print("Could not read image")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        try:
            landmarks = results.pose_landmarks.landmark
            landmark_data = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
            test_landmarks.append(landmark_data)
        except:
            continue

X = np.asarray(all_landmarks)
Y = np.asarray(all_labels)
X, Y = shuffle(X, Y)
y =[1]
X_test = np.asarray(test_landmarks)
X, X_test, Y, Y_test = train_test_split(
    X, 
    Y, 
    test_size=0.2, 
    random_state=42, 
    stratify=Y
)
clf1 = RandomForestClassifier()
clf1 = clf1.fit(X, Y)
clf2 = SVC(kernel="linear", C=1)
clf2 = clf2.fit(X, Y)
clf3 = tree.DecisionTreeClassifier()
clf3 = clf1.fit(X, Y)

clf= clf1
clf1 = clf1.predict(X_test)
print(clf1)
accuracy = accuracy_score(clf1, Y_test)
print(accuracy)
clf2=clf2.predict(X_test)
accuracy = accuracy_score(clf2, Y_test)
print(accuracy)
clf3=clf3.predict(X_test)
accuracy = accuracy_score(clf3, Y_test)
print(accuracy)
print(f"Shape of training data (X): {X.shape}")
print(f"Shape of training labels (Y): {Y.shape}")

if X_test.size > 0:
    print("Prediction on test data:", clf.predict(X_test))
else:
    print("No valid test data found. X_test is empty.")

cap = cv2.VideoCapture(path)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)
        
        try:
            landmarks = results.pose_landmarks.landmark
            landmark_data = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
            test_landmarks.append(landmark_data)
        except:
            continue
        
        X_test = np.asarray(test_landmarks)
        p = clf.predict(X_test)[-1]
        s = "at ease"
        if p == 1:
            s = "attention"
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        cv2.putText(image, str(s), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
