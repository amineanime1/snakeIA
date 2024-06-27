import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from IPython import display
from utils import load_record, load_total_games_played
import os

plot_counter = 0
total_games_played = load_total_games_played()

plt.ion()

def load_plot_counter():
    if os.path.exists('plot_counter.txt'):
        with open('plot_counter.txt', 'r') as file:
            return int(file.read())
    return 0

def save_plot_counter(counter):
    with open('plot_counter.txt', 'w') as file:
        file.write(str(counter))

def plot(scores, mean_scores, training_number):
    global plot_counter  # Accéder à la variable globale
    
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of games')
    plt.ylabel('Score')
    
    x_values = range(total_games_played, total_games_played + len(scores))
    mean_x_values = range(total_games_played, total_games_played + len(mean_scores))

    plt.plot(x_values, scores)
    plt.plot(mean_x_values, mean_scores)
    # Charger le record de tous les temps
    record_all_time = load_record()
    
    # Tracer la ligne droite représentant le record de tous les temps
    plt.plot([total_games_played, total_games_played + len(scores)], [record_all_time, record_all_time], 'r-', label='All-time record')
    
    record = max(scores)
    plt.plot([0, len(scores)], [record, record], 'g-', label='Current record')
    
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    plt.draw()  # draw the plot
    plt.pause(0.001)  # pause a bit so that the plot gets updated
    
    plt.xlim(left=total_games_played)
    
    interval = 100
    plt.xticks(range(total_games_played, total_games_played + len(scores), interval))
    
     # Sauvegarde de l'image dans le dossier 'images_exe'
    filename = f'images_exe/plot_{training_number:03d}.png'
    if os.path.exists(filename):
        os.remove(filename)  # Supprimer le fichier précédent s'il existe
    plt.savefig(filename)
