import random
import threading
import emoji

phrases = [
    f"\n[moose] I'm a moose and I'm hungry!",
    f"\n[moose] Bitch gimme some leaves {emoji.emojize(':red_heart:')}{emoji.emojize(':herb:')}",
    f"\n[moose] Where's my {emoji.emojize(':herb:')} like u know I am the largest of all the deer species in the world",
    f"\n[moose] Have you made enough {emoji.emojize(':money_with_wings:')} to sustain my diet bitch",
    f"\n[moose] I'm 1300 lbs sonnnnn hope you makin' dat {emoji.emojize(':money_with_wings:')}",
    f"\n[moose] Startup idea: Make {emoji.emojize(':money_bag:')}. Buy {emoji.emojize(':herb:')}. Feed moose. Profit.",
    f"\n[moose] I'm all in on ${emoji.emojize(':herb:')}LEAF",
    f"\n[moose] Do your ML models output {emoji.emojize(':herb:')} doe",
    f"\n[moose] The only currency I need is {emoji.emojize(':herb:')}",
    f"\n[moose] I LOVE {emoji.emojize(':herb:')}{emoji.emojize(':herb:')}{emoji.emojize(':herb:')}{emoji.emojize(':red_heart:')}",
    f"\n[moose] Buy and hold ${emoji.emojize(':herb:')}LEAF",
]
def moose_is_hungry():
    t = threading.Timer(10.0, moose_is_hungry)
    t.daemon = True
    t.start()
    print(random.choice(phrases))

