#!/usr/bin/env python

""" client.py - Echo client for sending/receiving C-like structs via socket
"""

import socket
import sys
import random
from ctypes import *
import numpy as np

class Payload(Structure):
    """ This class defines a C-like struct """
    _fields_ = [("id", c_uint32),
                ("counter", c_uint32),
                ("temp", c_float)]

def get_data():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_addr = ('localhost', 2300)
    collected = []

    if s is None:
        print("Error creating socket")

    try:
        s.connect(server_addr)
    except:
        print("ERROR: Connection to {a} refused".format(a=repr(server_addr)))
        sys.exit(1)

    try:
        
            print("")
            payload_out = Payload(1, 0, 0)
           
            s.send(payload_out)
            
            for _ in range(32*8):
                buff = s.recv(sizeof(Payload))
                payload_in = Payload.from_buffer_copy(buff)
                collected.append(np.round(payload_in.temp,3))
            


    finally:
        s.close()
        return collected
