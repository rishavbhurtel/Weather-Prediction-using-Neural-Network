import random
with open ("weather-data.txt") as f:
    lines = f.readlines()
random.shuffle(lines)
with open("shuffled.txt", "w") as f:
    f.writelines(lines)