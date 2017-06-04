import zmq
import json


class MirrorMessenger:

    def __init__(self, ip):

        self.ctx = zmq.Context.instance()
        self.s = self.ctx.socket(zmq.PUSH)
        self.url = 'tcp://{0}:5555'.format(ip)
        self.s.connect(self.url)

    def to_node(self, code, message):
        return json.dumps({code: message})

    def send_to_mirror(self, code, message):
        self.s.send(self.to_node(code, message))
