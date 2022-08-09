import random
import threading
import emoji

phrases = [f"\nI'm a moose and I'm hungry!",
           f"\nBitch gimme some leaves {emoji.emojize(':red_heart:')}{emoji.emojize(':herb:')}"]
def moose_is_hungry():
    t = threading.Timer(10.0, moose_is_hungry)
    t.daemon = True
    t.start()
    print(random.choice(phrases))

