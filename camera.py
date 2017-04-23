
def get_camera(use_usb_cam=True):
    try:
        if not use_usb_cam:
            import picam
            cam = picam.OpenCVCapture()
            cam.start()
            return cam
        else:
            raise Exception
    except Exception:
        import webcam
        return webcam.OpenCVCapture(device_id=0)