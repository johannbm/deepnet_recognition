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
import facenet_recognition
import face_landmarks
import opencv_modules
import user_recognizer
import nodejs_input


def shutdown(self, signum):
    nodejs_input.to_node("status", 'Shutdown: Cleaning up camera...')
    camera.stop()
    quit()


path_to_file = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
conf = json.load(open(path_to_file + '/conf.json'))

if conf["use_rpi_camera"]:
    camera = VideoStream(usePiCamera=conf["use_rpi_camera"] > 0).start()
    signal.signal(signal.SIGINT, shutdown)
    time.sleep(conf["camera_warmup_time"])
else:
    #camera = cv2.VideoCapture('http://{0}:2067/html/cam_pic_new.php'.format(conf["rpi_IP"]))
    camera = VideoStream(src='http://{0}:2067/html/cam_pic_new.php'.format(conf["rpi_IP"])).start()

user_rec = user_recognizer.UserRecognizer(conf)


def get_names(user_indexes):
    """
    Remember. It is 1-indexed
    :param user_indexes:
    :return:
    """
    users = conf["users"]
    names = []
    for i in user_indexes:
        if i < 1:
            names.append("{0} Unknown".format(i))
        else:
            names.append("{0} {1}".format(i, users[i-1]))
    return names

#cap.set(6, 5) cant remember what this does
#cap.set(cv2.cv.CV_CAP_PROP_FPS, 5)

#fl = face_landmarks.WinkClassifier()

# Initialize some variables
face_locations = []
face_names = []
process_this_frame = True
bg_sub_model = None
fps = 30
limit_fps = False
performance_stats = {}

print "Init recognition"
while True:
    # Grab a single frame of video
    start_time = time.time()
    frame = camera.read()
    #print camera.read()
    #ret, frame = camera.read()

    frame = imutils.resize(frame, width=500)

    if bg_sub_model is None:
        bg_sub_model = bgsub.BackgroundExtractor(frame, conf, path_to_file)

    # Only process every other frame of video to save time
    if process_this_frame:
        face_locations = bg_sub_model.detect_face(frame)
        face_names = user_rec.recognize_face(frame, face_locations)
        #fl.update_facial_landmarks(frame, face_locations)
        #fl.classify_current()
        user_rec.check_login()
        user_rec.check_logout()

        #print "landmark execution time: " + str(fl.performance_stats["Landmarks"])

    process_this_frame = not process_this_frame
 #   if conf["show_video"]["landmarks"]:
  #      fl.show_landmarks(frame)

    if conf["show_video"]["recognition"]:
        user_rec.show_recognized_face(frame, face_locations, get_names(face_names))
    else:
        print performance_stats
    execution_time = time.time() - start_time
    if limit_fps:
        if execution_time < 1/float(fps):
            time.sleep((1/float(fps)) - execution_time)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

