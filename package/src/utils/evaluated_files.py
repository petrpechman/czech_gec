import os

current_dir = os.getcwd()
filenames = os.listdir(current_dir)
filenames = [filename for filename in filenames if not filename.startswith('ckpt-')]
filenames = [filename for filename in filenames if filename.split('-')[0].isdigit()]

for filename in filenames:
    number = filename.split('-')[0]
    length = len(number) + 1 
    new_name = filename[length:] + '-' + number
    os.rename(filename, new_name)