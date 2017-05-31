import inspect
import json
import os
import time

import cv2
import imutils

import background_subtractor as bgsub
import user_recognizer

path_to_file = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
conf = json.load(open(path_to_file + '/conf.json'))

performance_stats = {}


def test_video(filename):
    cap = cv2.VideoCapture(filename)
    cap.set(6, 5)
    fps = 30
    limit_fps = False

    face_locations = []
    face_names = []

    dnr = user_recognizer.UserRecognizer(conf["recognition_algorithm"], conf)

    frame_counter = 0
    login_frames = []
    logout_frames = []
    bg_sub_model = None

    process_this_frame = True
    #performance_stats["found_faces"] = 0

    success = True

    while success:
        start_time = time.time()
        success, frame = cap.read()

        if not success:
            continue

        frame = imutils.resize(frame, width=500)

        if bg_sub_model is None:
            bg_sub_model = bgsub.BackgroundExtractor(frame, conf, path_to_file, conf["detection_algorithm"])

        if process_this_frame:
            face_locations = bg_sub_model.get_bounding_box(frame)
            #performance_stats["found_faces"] += len(face_locations)
            face_names = dnr.recognize_face(frame, face_locations)
            login = dnr.check_login()
            logout = dnr.check_logout()
            if login:
                login_frames.append((login, frame_counter))
            if logout:
                logout_frames.append((logout, frame_counter))

        if conf["show_video"]["recognition"]:
            dnr.show_recognized_face(frame, face_locations, face_names)

        execution_time = time.time() - start_time
        if limit_fps:
            if execution_time < 1 / float(fps):
                time.sleep((1 / float(fps)) - execution_time)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_counter += 1
    accumulate_performance_stats(bg_sub_model.get_performance_stats())
    accumulate_performance_stats(dnr.get_performance_stats())
    return {"logins": login_frames, "logouts": logout_frames}


def calculate_score(annotations, results):
    #(login-delay, incorrect-logins, premature-timeouts, login-takeovers)
    score = {}
    for key in annotations:
        annotations_data = annotations[key]
        results_data = results[key]
        incorrect_logins = 0
        premature_timeouts = 0
        login_delay = 0
        login_takeover = 0
        is_login_consumed = False
        print results_data
        for login in results_data["logins"]:
            user, frame_number = login
            if frame_number < annotations_data["login"]:
                incorrect_logins += 1
            elif frame_number < annotations_data["logout"]:
                if is_login_consumed:
                    if user == annotations_data["user"]:
                        premature_timeouts += 1
                    else:
                        login_takeover += 1
                else:
                    if user == annotations_data["user"]:
                        login_delay = frame_number - annotations_data["login"]
                        is_login_consumed = True
                    else:
                        incorrect_logins += 1
        score[key] = (login_delay, incorrect_logins, premature_timeouts, login_takeover)
    return score


def accumulate_performance_stats(stats):
    for key in stats:
        if performance_stats.has_key(key):
            performance_stats[key].append(stats[key])
        else:
            performance_stats[key] = []


def summarize_score(score, conf):
    total_login_delay = 0
    total_incorrect = 0
    total_premature = 0
    total_takeover = 0
    for key in score:
        total_login_delay += score[key][0]
        total_incorrect += score[key][1]
        total_premature += score[key][2]
        total_takeover += score[key][3]

    score_size = float(len(score))
    print "-" * 20 + " Configurations " + "-"*20
    print "Detection algorithm: {0}".format(bgsub.BackgroundExtractor.get_algorithm_text(conf["detection_algorithm"]))
    print "Recognition algorithm: {0}".format(user_recognizer.UserRecognizer.get_algorithm_text(conf["recognition_algorithm"]))
    print "Consecutive detection limit: {0}".format(conf["consecutive_detections"])

    print "-" * 20 + " Scores " + "-"*20
    print "Average login delay {0}".format(total_login_delay/score_size)
    print "Total incorrects {0}".format(total_incorrect)
    print "Total premature logins {0}".format(total_premature)
    print "Total takeovers {0}".format(total_takeover)

    print "-" * 20 + " Time Stats " + "-" * 20
    for key in performance_stats:
        if type(performance_stats[key]) is list:
            print "{0} average time: {1}".format(key, sum(performance_stats[key]) / float(len(performance_stats[key])))
    print "Total running time: {0}".format(performance_stats["total_time"])


annotations = json.load(open(path_to_file + '/annotations.json'))
#print annotations
results = {}
start_time = time.time()
for key in annotations:
    results[key] = test_video('../test_data/training_videos/' + key)
performance_stats["total_time"] = time.time() - start_time
score = calculate_score(annotations, results)
summarize_score(score, conf)
