import os
import numpy as np

def list_avg(l):
    """
    Calculates the average value of given list
    :param l: list of numbers
    :return: average value
    """
    return sum(l) / float(len(l)) if len(l) > 0 else 0


def are_all_elements_equal(l):
    """
    checks if all elements in the list are equal
    :param l: list to check
    :return: True if all elements are equal, otherwise False
    """
    return l[1:] == l[:-1]


def get_sorted_directory(folder):
    """
    Creates a sorted list of basenames in directory
    :param folder: folder to sort
    :return: sorted list of basenames in folder
    """
    return sorted(os.listdir(folder))


def most_common_element(lst):
    """
    Finds the most common element in a list. Break ties randomly
    :param lst: list of elements
    :return: most commong element
    """
    return max(set(lst), key=lst.count)


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
    """
    Converts an opencv location(s) to dlib location. Ensure all numbers are of type numpy.int64
    :param location: list of location(s) in opencv format (x, y, w, h)
    :return: list of locations in dlib format (top, right, bottom , left)
    """
    return [_opencv_to_dlib(x) for x in location]


def _opencv_to_dlib(location):
    if len(location) == 0:
        return ()
    x, y, w, h = location.astype(np.int64)
    return y, w + x, y + h, x
