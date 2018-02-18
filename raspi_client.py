"""
client.py
send photo to server computer every fixed time
"""
# socket
import socket               # Import socket module
import picamera
from time import sleep
from fractions import Fraction


shut_sp = 400000
iso_val = 100

host = "XXX.XXX.XX.XX" # GPU machine IP address
port = 55000                # Reserve a port for your service.

with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)
    # Set a framerate of 1/6fps, then set shutter
    # speed to 6s and ISO to 800
    while True:
        # --------------- Take Photo using raspi
        camera.framerate = Fraction(1, 6)
        camera.shutter_speed = shut_sp
        camera.exposure_mode = 'off'
        camera.iso = iso_val
        camera.capture('latest_pic.jpg')
        print("captured real ")

        # ---------------- Send captured image using socket

        s = socket.socket()         # Create a socket object
        s.connect((host, port))
        print("succesfully connectod to host")
        f = open('latest_pic.jpg','rb')
        count = 1
        l = f.read(1024)
        while (l):
            count = count + 1
            s.send(l)
            l = f.read(1024)
        f.close()
        print count
        s.close()

        # ---------------- Wait for next loop
        print("wait 20 sec")
        sleep(3)
