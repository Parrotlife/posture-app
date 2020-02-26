#!/usr/bin/env python

""" client.py - Echo client for sending/receiving C-like structs via socket
References:
- Ctypes fundamental data types: https://docs.python.org/2/library/ctypes.html#ctypes-fundamental-data-types-2
- Ctypes structures: https://docs.python.org/2/library/ctypes.html#structures-and-unions
- Sockets: https://docs.python.org/2/howto/sockets.html
"""

import socket
import sys
import random
from ctypes import *
import numpy as np


""" This class defines a C-like struct """
class Payload(Structure):
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
        # print("Connected to {a}".format(a=repr(server_addr)))
    except:
        print("ERROR: Connection to {a} refused".format(a=repr(server_addr)))
        sys.exit(1)

    try:
        
            print("")
            payload_out = Payload(1, 0, 0)
            # print("Sending id={a:5d}, counter={b:5d}, temp={c:.2f}".format(a=payload_out.id,
            #                                             b=payload_out.counter,
            #                                             c=payload_out.temp))
            nsent = s.send(payload_out)
            # Alternative: s.sendall(...): coontinues to send data until either
            # all data has been sent or an error occurs. No return value.
            # print("Sent {a:5d} bytes".format(a=nsent))
            
            
            for _ in range(32*8):
                buff = s.recv(sizeof(Payload))
                payload_in = Payload.from_buffer_copy(buff)
                # print("Received id={a:5d}, counter={b:5d}, temp={c:.2f}".format(a=payload_in.id,
                #                                         b=payload_in.counter,
                #                                         c=payload_in.temp))
                collected.append(np.round(payload_in.temp,3))
            


    finally:
        s.close()
        return collected
