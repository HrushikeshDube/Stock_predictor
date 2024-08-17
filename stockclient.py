import socket
import time


class StockPSocket:
    def __init__(self, host, port):
        """
         Constructor: creates a socket object and verifies the server is listening
         Args:
            self: calling object
            host: string ip address of server
            port: int port number on server
         Returns:
            void
         """
        # creates socket
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # IPv4 address of server
        self.host = host
        # port of server
        self.port = port
        # makes connection
        try:
            print(self.host, self.port)
            self.s.connect((self.host, self.port))
        except OSError:
            print('Error occurred when connecting to server')
            raise ConnectionRefusedError()

    # validates stock ticker input
    def validate(self, tkr):
        """
         Verifies the tkr is valid
         Args:
            self: calling object
            tkr: string stock ticker + 'v'
         Returns:
            value returned from check_validation(): 'success' or 'error'
         """
        self.s.sendall(tkr.encode('UTF-8'))
        return self.check_validation()

    def check_validation(self):
        """
         Receives data from server concerning tkr validation
         Args:
            self: calling object
         Returns:
            inp: 'success' or 'error'
         """
        inp = self.s.recv(1024).decode('UTF-8')
        return inp

    # requests data from server
    def send_request(self, request):
        """
         Requests for prediction and graph to be created
         Args:
            self: calling object
            request: string ticker + 'p' or ticker + 'g'
         Returns:
            void
         """
        self.s.sendall(request.encode('UTF-8'))

    # receives prediction
    def rec_pred(self):
        """
         Receives prediction from server
         Args:
            self: calling object
         Returns:
            predicted value stored in pred as a string
         """
        pred = self.s.recv(1024).decode('UTF-8')
        return pred

    # receives data from server
    def receive(self):
        """
         Receives a binary .png file ans stores writes it to imgFile.png
         Args:
            self: calling object
         Returns:
            void
         """
        self.s.setblocking(1)
        data = []
        inp = ''
        begin = time.time()
        while True:
            if time.time() - begin > 3:
                break
            try:
                inp = self.s.recv(4096)
                if not inp: break
                if inp:
                    data.append(inp)

                    begin = time.time()
                else:
                    time.sleep(0.1)

            except Exception as e:
                print(str(e))
                break
                pass
        rec = b''.join(data)
        with open('imgFile.png', 'wb') as f:
            f.write(rec)

    def close(self):
        """
         Closes socket
        """
        self.s.close()
