import os, time

filenames = ['Fireworks_1.txt', 'Fireworks_2.txt', 'Fireworks_3.txt']

def animation_run(filenames, delay = .5, repetition = 10):
    os.system('clear')
    frames = []

    for name in filenames:
        with open(name, 'r', encoding='utf8') as f:
            frames.append(f.readlines())

    for i in range(repetition):
        for frame in frames:
            print("".join(frame))
            time.sleep(delay)
            os.system('clear')

    return

animation_run(filenames)

