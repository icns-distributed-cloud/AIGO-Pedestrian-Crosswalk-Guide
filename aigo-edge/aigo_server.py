import socket 
import cv2
import numpy
from select import *
from _thread import *

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def threaded(client_socket, addr): 

    print('Connected by :', addr[0], ':', addr[1]) # addr[0] 클라이언트 주소, addr[1] 접속한 클라이언트 포트

    n = 0
    while True: 

        try:
           
            data = client_socket.recv(1024)        #client로 부터 1을 받음

            if not data: 
                print('Disconnected by ' + addr[0],':',addr[1])   
                break

            message = '2'
            client_socket.send(message.encode())

            length = recvall(client_socket,16)
            
            stringData = recvall(client_socket, int(length))
            data = numpy.frombuffer(stringData, dtype="uint8")

            decimg = cv2.imdecode(data,1)
            cv2.imshow('Image', decimg)
            cv2.imwrite('images/test.jpg',decimg)
            n += 1

            key = cv2.waitKey(1)
            if key == 27:                #esc누르면 종료
                break



        except ConnectionResetError as e:

            print('Disconnected by ' + addr[0],':',addr[1])
            break
             
    client_socket.close() 


HOST = '127.0.0.1'
PORT = 9999

server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('', PORT))

server_socket.listen()

print('server start')
socketList = [server_socket]

while True:

    print('wait')

    try:
        read_socket, write_socket, error_socket = select(socketList, [], [], 1)
    
        for sock in read_socket:
            if sock == server_socket:
                client_socket, addr = server_socket.accept()
                start_new_thread(threaded, (client_socket, addr,))
    except KeyboardInterrupt:
        break

server_socket.close() 