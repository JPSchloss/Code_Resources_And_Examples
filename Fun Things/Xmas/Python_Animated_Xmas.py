import os, time

# Set the ASCII Art Filenames
filenames = ['Merry_Xmas_1.txt', 'Merry_Xmas_2.txt', 'Merry_Xmas_3.txt', 'Merry_Xmas_4.txt', 
             'Merry_Xmas_5.txt', 'Merry_Xmas_6.txt', 'Merry_Xmas_7.txt', 'Merry_Xmas_8.txt', 
             'Merry_Xmas_9.txt', 'Merry_Xmas_10.txt', 'Merry_Xmas_11.txt', 'Merry_Xmas_12.txt', 'Merry_Xmas_13.txt']

# Define function for animation
def animation_run(filenames, delay = .5, repetition = 1):

    # Clear the screen and create an object to hold the ASCII files.
    os.system('clear')
    frames = []

    # Loop through the files and load them into the environment. 
    for name in filenames:
        with open(name, 'r', encoding='utf8') as f:
            frames.append(f.readlines())

    # Loop through a range and each image to create the animation. 
    #for i in range(repetition):
    for frame in frames:
        print("".join(frame))
        time.sleep(delay)
        os.system('clear')

    print("".join(frames[-1]))

    return

animation_run(filenames)