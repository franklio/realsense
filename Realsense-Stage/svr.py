#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## Ref : https://shengyu7697.github.io/python-socket/

import socket
import json

HOST = '127.0.0.1'
#HOST = '140.118.127.144'

PORT = 8000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

## Domain
## 1 AF_INET         IPv4因特網域 （兩台主機透過網路進行資料傳輸）
## 2 AF_INET6        IPv6因特網域 （兩台主機透過網路進行資料傳輸）
## 3 AF_UNIX         Unix域      （程序與程序間的傳輸）

## Type
## 1 SOCK_STREAM     序列化的連接導向通訊。對應的protocol為TCP。
## 2 SOCK_DGRAM      提供的是一個一個的資料包(datagram)。對應的protocol為UDP

server.bind((HOST, PORT))
server.listen(10)

while True:
    conn, addr = server.accept()
    clientMessage = str(conn.recv(1024), encoding='utf-8')

    print('Client message is:', clientMessage)
    print(addr)
    #check = json.loads(clientMessage)
    #print(check["hand_left_keypoints_3d"])

    serverMessage = 'I\'m here!'
    conn.sendall(serverMessage.encode())
    conn.close()
