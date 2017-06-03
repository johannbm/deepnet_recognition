
def list_avg(l):
    return sum(l) / float(len(l)) if len(l) > 0 else 0


def basic_crop_dlib(image, face_location):
    """
    Crops the image around the given location
    :param image: original image
    :param face_location: A rectangle in dlib-format
    :return: cropped image
    """
    top, right, bottom, left = face_location
    return image[top:bottom, left:right]


def convert_dlib_location_to_opencv(location):
    """
    Converts a dlib rect tuple to opencv rect tuple
    :param location: dlib rect (top, right, bottom, left)
    :return: opencv rect (x, y, w, h)
    """
    top, right, bottom, left = location
    return left, top, right-left, bottom-top


def convert_opencv_location_to_dlib(location):
    return [_opencv_to_dlib(x) for x in location]


def _opencv_to_dlib(location):
    if len(location) == 0:
        return ()
    x, y, w, h = location
    return y, w + x, y + h, x
