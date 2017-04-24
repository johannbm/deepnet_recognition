import face_recognition
import cv2
import time
import background_subtractor as bgsub
import imutils
import json
import sys
import inspect
import os
import signal
from imutils.video import VideoStream


def to_node(type, message):
    # convert to json and print (node helper will read from stdout)
    try:
        print(json.dumps({type: message}))
    except Exception:
        pass
    # stdout has to be flushed manually to prevent delays in the node helper communication
    sys.stdout.flush()


def shutdown(self, signum):
    to_node("status", 'Shutdown: Cleaning up camera...')
    camera.stop()
    quit()

signal.signal(signal.SIGINT, shutdown)

path_to_file = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

conf = json.load(open(path_to_file + '/conf.json'))

camera = VideoStream(conf["use_rpi_camera"] > 0).start()
time.sleep(conf["camera_warmup_time"])

# Load a sample picture and learn how to recognize it.
johannes_image = face_recognition.load_image_file(path_to_file + "/images/johannes.jpg")
mathias_image = face_recognition.load_image_file(path_to_file + "/images/mathias.jpg")
ingunn_image = face_recognition.load_image_file(path_to_file + "/images/ingunn.jpg")
jessie_image = face_recognition.load_image_file(path_to_file + "/images/jessie.jpg")

johannes_face_encoding = face_recognition.face_encodings(johannes_image)[0]
mathias_face_encoding = face_recognition.face_encodings(mathias_image)[0]
ingunn_face_encoding = face_recognition.face_encodings(ingunn_image)[0]
jessie_face_encoding = face_recognition.face_encodings(jessie_image)[0]
faces = [johannes_face_encoding, mathias_face_encoding, ingunn_face_encoding, jessie_face_encoding]


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
bg_sub_model = None
performance_stats = {}
time_since_Known_face = 0
current_user = None
recent_detections = [-1]*conf["consecutive_detections"]
detection_index = 0


def get_face_locations_dlib(frame):
    return face_recognition.face_locations(frame)


def get_face_locations_bgsub(frame):
    return bg_sub_model.getBoundingBox(frame)


def get_face_encoding(frame, face_locations, debug=True):
    time_start = time.time()
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    if debug:
        time_end = time.time()
        performance_stats["Encoding"] = time_end-time_start

    return face_encodings


def get_face_locations(frame, debug=True):
    time_start = time.time()
    face_areas = get_face_locations_bgsub(frame)

    if debug:
        time_end = time.time()
        performance_stats["Detection"] = time_end-time_start
    print face_areas
    return face_areas


def get_face_match(faces, face_encoding, debug=True):
    time_start = time.time()
    match = face_recognition.compare_faces(faces, face_encoding)

    if debug:
        time_end = time.time()
        performance_stats["Matching"] = time_end-time_start

    return match


def get_facial_landmarks(face, location, debug=True):
    time_start = time.time()
    landmarks = face_recognition.face_landmarks(face, location)

    if debug:
        time_end = time.time()
        performance_stats["Landmarks"] = time_end-time_start

    return landmarks


def update_detection_list(index):
    global detection_index
    recent_detections[detection_index] = index
    detection_index = (detection_index + 1) % conf["consecutive_detections"]


def check_login():
    global recent_detections, current_user
    if recent_detections[0] != -1 and recent_detections[1:] == recent_detections[:-1] and current_user != recent_detections[0]:
        current_user = recent_detections[0]
        to_node("login", {"user": current_user+1})


def check_logout():
    global current_user, recent_detections
    if time.time() - time_since_Known_face > conf["logout_time"] and current_user is not None:
        to_node("logout", {"user": current_user+1})
        recent_detections = [-1]*conf["consecutive_detections"]
        current_user = None

print "Init recognition"
while True:
    # Grab a single frame of video
    frame = camera.read()
    frame = imutils.resize(frame, width=500)

    if bg_sub_model is None:
        bg_sub_model = bgsub.BackgroundExtractor(frame, conf, path_to_file)

    # Only process every other frame of video to save time
    if process_this_frame:
        face_locations = get_face_locations(frame)
        face_encodings = get_face_encoding(frame, face_locations)
        landmarks = get_facial_landmarks(frame, face_locations)
        print landmarks

        face_names = []
        for face_encoding in face_encodings:
            match = get_face_match(faces, face_encoding)

            name, index = "Unknown", -1

            for i in range(len(match)):
                if match[i]:
                    time_since_Known_face = time.time()
                    index = i

            update_detection_list(index)
            face_names.append(conf["users"][index])
        print face_names    
        check_login()
        check_logout()

    process_this_frame = not process_this_frame

    if conf["show_video"]["recognition"]:
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(frame, "Detection: " + str(round(performance_stats["Detection"], 5)), (10, 20), font, 1.0, (255, 255, 255), 1)
            cv2.putText(frame, "Encoding: " + str(round(performance_stats["Encoding"], 5)), (10, 40), font, 1.0, (255, 255, 255), 1)
            cv2.putText(frame, "Matching: " + str(round(performance_stats["Matching"], 5)), (10, 60), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print performance_stats
# Release handle to the webcam
#video_capture.release()
#cv2.destroyAllWindows()
