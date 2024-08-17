# Developed by: Jordan Davis and Ryan Harris
#
# Emails: davisja2023@mountunion.edu and rh297019@ohio.edu
#
# Description:This program has a user-facing GUI that interacts
# with a server. The server predicts future stock prices using
# an Long-Short-Term-Memory (LSTM) network. The training of
# the model takes place on the server and the result of the
# prediction is sent back to the client with as little latency
# as possible.
#
# NOTE: The server object was also designed and implemented
# by Jordan Davis and Ryan Harris.
from tkinter import *
from tkinter import messagebox
from tkinter.ttk import Button, Label, Entry
from stockclient import StockPSocket
from PIL import Image, ImageTk


class MainGUI:
    def __init__(self, master):
        """
         Constructor: Creates the GUI and creates a socket
         Args:
            self: calling object
            master: parent widget
         Returns:
            void
         """
        # sets host and port of server
        self.host = '127.0.0.1'
        self.port = 9998

        # sets up empty var for ticker
        self.tkr = ''

        # sets up prediction variable
        self.prediction = 0

        self.master = master
        master.title('Stock Predictor')
        master.geometry('800x600')

        # creates label
        self.lbl = Label(master, text='Input Stock Ticker')
        self.lbl.pack(side=TOP)

        # creates text field
        self.txt = Entry(master, width=10)
        self.txt.pack(side=TOP)

        # creates submission button
        self.btn = Button(master, text='Submit', command=self.clicked_tkr)
        self.btn.pack(side=TOP)

        # creates prediction explicit label
        self.p_lbl = Label(master, text='Prediction(1 day in the future): ')
        self.p_lbl.pack(side=TOP)

        # creates prediction output label
        self.p_out_lbl = Label(master, text=self.get_prediction())
        self.p_out_lbl.config(font=17)
        self.p_out_lbl.pack(side=TOP)

    def disp_graph(self, wait=False):
        """
         Displays the image stored as imgFile.png to the GUI
         Args:
            self: calling object
            wait: boolean True if waiting for image False if not
         Returns:
            void
         """
        try:
            if wait:
                img = Image.open('waitImg.png')
            else:
                img = Image.open('imgFile.png')
            render = ImageTk.PhotoImage(img)
            # create image label
            img = Label(self.master, image=render)
            img.image = render
            img.place(x=95, y=110)
        except OSError as OSE:
            messagebox.showinfo('Graph Retrieval Failed', 'The graph requested was not received correctly. Please '
                                                          'submit the request again.')

    def clicked_tkr(self):
        """
         Gathers the input from self.txt, verifies it, then requests a prediction and a graph. It then finally calls
         disp_graph(). Also handles server offline and invalid input cases.
         Args:
            self: calling object
         Returns:
            void
         """
        ticker = ""
        try:
            if self.txt.get().strip():
                ticker = self.txt.get().strip()
            else:
                raise ValueError('Empty string')

            self.update_prediction_out('Please wait. This could take up to 1 minute.')

            client_socket = StockPSocket(self.host, self.port)
            if client_socket.validate(ticker + 'v') == 'error':
                client_socket.close()
                self.update_prediction_out('Invalid input')
                raise ValueError()
            else:
                self.tkr = ticker
                self.disp_graph(wait=True)

                # gets predicted value of stock
                client_socket = StockPSocket(self.host, self.port)
                client_socket.send_request(self.tkr + 'p')
                prediction = client_socket.rec_pred()
                client_socket.close()
                self.update_prediction_out(prediction)  # after Ryan finishes lstm prediction

                # gets graph of stock
                client_socket = StockPSocket(self.host, self.port)
                client_socket.send_request(self.tkr + 'g')
                client_socket.receive()
                client_socket.close()
                self.disp_graph()

        except ValueError as VE:
            messagebox.showinfo('Invalid Ticker', 'Please enter a proper stock ticker. If you are trying to get data '
                                                  'about a market index prefix the ticker with a ^ (ie. ^dji).')
        except ConnectionRefusedError as CRE:
            messagebox.showinfo('Server Offline', 'Sorry for the inconvenience, but this service is not available '
                                                  'right now.')

    def set_prediction(self, prediction):
        """
         Updates the prediction variable
         Args:
            self: calling object
            prediction: int updated value
         Returns:
            void
         """
        self.prediction = prediction

    def get_prediction(self):
        """
         Returns the current value stored in prediction
         Args:
            self: calling object
         Returns:
            prediction: int
         """
        return self.prediction

    def update_prediction_out(self, prediction):
        """
         Updates the prediction label
         Args:
            self: calling object
            prediction: int value for updated presiction
         Returns:
            void
         """
        self.set_prediction(prediction)
        self.p_out_lbl['text'] = self.get_prediction()
        self.p_out_lbl.update()


root = Tk()
root.resizable(width=False, height=False)
my_gui = MainGUI(root)
root.mainloop()
