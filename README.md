# Stock-Predictor
## Overview
This program features a graphical user interface that connects to a custom server to fetch a predicted value for the inputted stock. It uses a LSTM to make predictions and learns by using the mean squared error loss function. Predictions are graphed for the inputted stock upon prediction completion. 
## Project Download Instructions
- Clone this repository
- Execute the <code>pip install -r requirements.txt</code> command in your terminal on your local machine(sudo or administrator privileges might be required)
## Client Execution Instructions
### Option 1
- Windows: <code>python mainGUI.py</code>
- macOS and Linux: <code>python3 mainGUI.py</code>
### Option 2
- Double click on mainGUI.py to execute using the Python 3 interpreter
## Server Execution Instructions
- Important: server.py file must be executed on weberkcudafac, since this machine has a static ip and that ip is the default for the server
- Note: after cloning the repo onto weberkcudafac, the following command needs executed<code>pip install -r requirements.txt</code>
- To execute: enter the terminal and input <code>python3 server.py</code>
- To kill: enter <code>ctrl+c</code>


