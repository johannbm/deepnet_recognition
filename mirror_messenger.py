import zmq
import json


class MirrorMessenger:

    def __init__(self, ip):

        self.ctx = zmq.Context.instance()
        self.s = self.ctx.socket(zmq.PUSH)
        self.url = 'tcp://{0}:5555'.format(ip)
        self.s.connect(self.url)

    def to_node(self, type, message):
        return json.dumps({type: message})

    def send_to_mirror(self, type, message):
        self.s.send(self.to_node(type, message))
