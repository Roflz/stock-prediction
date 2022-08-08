import threading


def moose_is_hungry():
    t = threading.Timer(10.0, moose_is_hungry)
    t.daemon = True
    t.start()
    print("I'm a moose and I'm hungry!")


moose_is_hungry()
