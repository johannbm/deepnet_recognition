import mirror_messenger

m = mirror_messenger.MirrorMessenger()

while True:
    msg = raw_input("enter text")
    m.send_to_mirror("login", {"user": 1})
