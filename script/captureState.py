capture_status = {"complete": False}
traininge_status = {"complete": False}

def set_capture_complete(status):
    global capture_status
    capture_status["complete"] = status

def is_capture_complete():
    return capture_status["complete"]

def set_training_complete(status):
    global traininge_status
    traininge_status["complete"] = status

def is_training_complete():
    return traininge_status["complete"]