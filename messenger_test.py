import mirror_messenger

ip = "enter test ip here"

m = mirror_messenger.MirrorMessenger(ip)

while True:
    msg = raw_input("enter text")
    m.send_to_mirror("login", {"user": 1})
