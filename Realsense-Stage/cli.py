#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## Ref : https://shengyu7697.github.io/python-socket/

import socket

#HOST = '0.0.0.0'
HOST = '140.118.127.144'
PORT = 8000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

outdata = 'hello tcp'
print('send: ' + outdata)
s.send(outdata.encode())

indata = s.recv(1024)
print('recv: ' + indata.decode())

s.close()