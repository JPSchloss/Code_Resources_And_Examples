import os, time

# Set the ASCII Art Filenames
filenames = ['Fireworks_1.txt', 'Fireworks_2.txt', 'Fireworks_3.txt']

# Define function for animation
def animation_run(filenames, delay = .5, repetition = 10):

    # Clear the screen and create an object to hold the ASCII files.
    os.system('clear')
    frames = []

    # Loop through the files and load them into the environment. 
    for name in filenames:
        with open(name, 'r', encoding='utf8') as f:
            frames.append(f.readlines())

    # Loop through a range and each image to create the animation. 
    for i in range(repetition):
        for frame in frames:
            print("".join(frame))
            time.sleep(delay)
            os.system('clear')

    return

animation_run(filenames)

