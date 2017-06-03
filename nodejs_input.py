import json
import sys


def to_node(code, message):
    # convert to json and print (node helper will read from stdout)
    try:
        print(json.dumps({code: message}))
    except Exception:
        pass
    # stdout has to be flushed manually to prevent delays in the node helper communication
    sys.stdout.flush()
