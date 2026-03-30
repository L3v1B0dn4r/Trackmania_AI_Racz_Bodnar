import pydirectinput

def accelerate():
    pydirectinput.keyDown('up')

def brake():
    pydirectinput.keyDown('down')

def left():
    pydirectinput.keyDown('left')

def right():
    pydirectinput.keyDown('right')

def release_all():
    pydirectinput.keyUp('up')
    pydirectinput.keyUp('down')
    pydirectinput.keyUp('left')
    pydirectinput.keyUp('right')