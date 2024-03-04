import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Load data from JSON file
with open('/home/marcuswrrn/Projects/Machine_Learning/NLP/Codenames/recorded_results/model_output_standard.json', 'r') as file:
    data = json.load(file)

# Function to plot the data
def plot_data(data, title):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))  # Create a figure with subplots
    fig.suptitle(title, fontsize=16)

    # Extract data for plotting
    for i, (key, output) in enumerate(data.items()):
        avg_targets = output['Avg Targets']
        neutral = output['Neutral']
        negative = output['Negative']
        assassin = output['Assassin']
        trials = [i for i in range(len(avg_targets))]


        expected_neg_line = [9/16 for _ in range(len(avg_targets))]
        expected_neut_line = [6/16 for _ in range(len(avg_targets))]

        # Plotting
        #ax[i // 2, i % 2].plot(avg_targets, neutral, label='neutral', marker='o', linestyle='-', color="blue")
        #ax[i // 2, i % 2].plot(avg_targets, negative, label='negative', marker='s', linestyle='-', color="orange")

        # Expected rates with dotted lines
        #ax[i // 2, i % 2].plot(avg_targets, expected_neg_line, label='expected negative rate', linestyle=':', color="orange")
        #ax[i // 2, i % 2].plot(avg_targets, expected_neut_line, label='expected neutral rate', linestyle=':', color="blue")
        ax[i // 2, i % 2].plot(avg_targets, assassin, label='Assassin', marker='^')
        # ax[i // 2, i % 2].set_title(key)
        ax[i // 2, i % 2].set_xlabel('Epoch')
        ax[i // 2, i % 2].set_ylabel('Selection Rate')

        legend_lines = [Line2D([0], [0], color='blue', lw=2, label='neutral rate'),
                        Line2D([0], [0], color='orange', lw=2, label='negative rate')]
        
        ax[i // 2, i % 2].legend(handles=legend_lines)
    print()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Plotting the data
for weight_type, content in data.items():
    plot_data(content, weight_type)