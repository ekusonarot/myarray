import random
for h in range(100):
    for w in range(100):
        if random.random() < 0.5:
            print(".", end="")
        else:
            print("#", end="")
    print("")