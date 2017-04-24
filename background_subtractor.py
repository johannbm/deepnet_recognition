import cv2
import time
import imutils
import json
import os
import inspect
from imutils.video import VideoStream

class BackgroundExtractor:

    def __init__(self, firstFrame, conf, path_to_file):
        frame = imutils.resize(firstFrame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        self.avg = gray.copy().astype("float")
        self.current_frame = self.avg
        self.previous_positive_detection = time.time()

        self.negative_seconds_limit = conf["consecutive_negative_seconds_limit"]
        self.min_area = conf["min_area"]
        self.show_feed = conf["show_video"]
        self.is_dynamic = conf["dynamic_background"]
        self.face_cascade = cv2.CascadeClassifier(os.path.join(path_to_file, conf["cascade_path"]))


    def getPotentialRegions(self, frame):
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        self.current_frame = gray

        if self.show_feed["blur"]:
            cv2.imshow('Blurred', gray)

        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg))

        if self.show_feed["average"]:
            cv2.imshow('Average', cv2.convertScaleAbs(self.avg))

        thresh = cv2.threshold(frameDelta, 15, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        if self.show_feed["contour"]:
            cv2.imshow("Countor", thresh)

        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

        areas = [x for x in cnts if cv2.contourArea(x) > self.min_area]
        
        cropped_images = []
        for a in areas:
            x, y, w, h = cv2.boundingRect(a)
            cropped_images.append((frame[y:y + h, x:x + w], (x, y, w, h)))

            if self.show_feed["detection"]:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.imshow('Video CONTOUR', frame)

        return cropped_images

    def getBoundingBox(self, frame):
        potentialAreas = self.getPotentialRegions(frame)
        bounding_boxes = []

        for area, bounding_box in potentialAreas:
            faces = self.face_cascade.detectMultiScale(
                area,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )
            #update background model if noe faces found
            if len(faces) == 0 and self.is_dynamic:
                self.accumulate_background()
            else:
                print "face detected"
            
            bounding_boxes.extend(self.get_face_bounds(faces, bounding_box))
            
        return bounding_boxes

    def accumulate_background(self):
        if time.time()-self.previous_positive_detection > self.negative_seconds_limit:
            cv2.accumulateWeighted(self.current_frame, self.avg, 0.01)
            #print "updating"
            #self.avg = self.current_frame


    def get_face_bounds(self, faces, bounding_box):
        bounding_boxes = []
        for face in faces:
            (x1, y1, w1, h1) = face
            x, y, w, h = bounding_box
            top = y+y1
            right = x+x1+w1
            bottom = y+y1+h1
            left = x+x1
            bounding_boxes.append((top, right, bottom, left))
            self.previous_positive_detection = time.time()
        return bounding_boxes



if __name__ == "__main__":
    conf = json.load(open('conf.json'))
    path_to_file = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    video_capture = VideoStream(usePiCamera=False > 0).start()
    time.sleep(2.0)

    bgsub = BackgroundExtractor(video_capture.read(), conf, path_to_file)
    face_cascade = cv2.CascadeClassifier('cascades/lbpcascade_frontalface.xml')
    debug=True
    try:
        while True:
            frame = video_capture.read()
            frame = imutils.resize(frame, width=500)

            potentialAreas = bgsub.getPotentialRegions(frame)

            face_locations = bgsub.getBoundingBox(frame)

            if debug:
                regions = bgsub.getBoundingBox(frame)
                for x,y,w,h in regions:
                    cv2.rectangle(frame, (h, x), (y, w), (100, 0, 100), 2)

                print "negative frames count: " + str(time.time()-bgsub.previous_positive_detection)



            # all_faces = []
            # for area, bounding_box in potentialAreas:
            #     faces = face_cascade.detectMultiScale(
            #         area,
            #         scaleFactor=1.1,
            #         minNeighbors=5,
            #         minSize=(30, 30),
            #         flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            #     )
            #     if conf["show_video"]["detection"]:
            #         for face in faces:
            #             (x1, y1, w1, h1) = face
            #             x, y, w, h = bounding_box
            #             cv2.rectangle(frame, (x+x1, y+y1), (x+x1 + w1, y+y1 + h1), (0, 255, 0), 2)
            #             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.imshow('Face detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass


