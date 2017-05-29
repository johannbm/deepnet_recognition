import zmq
import json


ctx = zmq.Context.instance()
s = ctx.socket(zmq.PUSH)
url = 'tcp://192.168.108:5555'
s.connect(url)
current_user = 3

def to_node(type, message):
    return json.dumps({type: message})


while True:
    msg = to_node("login", {"user": 1})
    res = raw_input("click to continue")
    #msg = raw_input("msg > ")
    s.send(msg)
    if msg == 'quit':
        break
