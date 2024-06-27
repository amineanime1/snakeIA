import os

DATA_DIR = 'data_trainings'

def load_data(filename, default=0):
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            return int(file.read())
    return default

def save_data(filename, data):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w') as file:
        file.write(str(data))

def load_total_trainings():
    return load_data('total_trainings.txt')

def save_total_trainings(total_trainings):
    save_data('total_trainings.txt', total_trainings)

def load_total_games_played():
    return load_data('total_games.txt')

def save_total_games_played(total_games_played):
    save_data('total_games.txt', total_games_played)

def load_record():
    return load_data('record.txt')

def save_record(record):
    save_data('record.txt', record)
    
def empty_images_folder():
    folder_path = 'images_exe'
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("Images folder emptied successfully.")
    else:
        print("The images folder does not exist or is not a directory.")