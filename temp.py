import re
from tqdm import tqdm

ansi_escape_pattern = re.compile(r'\x1b\[[0-9;]*m')


with open('/mnt/home/xxx/pycharm_workspace/adrs/log/0812-3/log.txt', 'r') as file:
    log_data = file.readlines()

info_dict = {
    'loss': {},
    'map': {}
}

epoch = -1
for line in tqdm(log_data, desc='analysing'):

    if 'epoch' in line and 'step 0/' in line and 'loss' in line:
        epoch += 1

    if 'epoch' in line and 'step ' in line and 'loss' in line:

        loss = float(ansi_escape_pattern.sub('', line).split('loss ')[1].strip())
        if f'epoch{epoch}' in info_dict['loss']:
            info_dict['loss'][f'epoch{epoch}'].append(loss)
        else:
            info_dict['loss'][f'epoch{epoch}'] = [loss]

    if 'Mean Average Precision (mAP)' in line:
        map = float(ansi_escape_pattern.sub('', line).split('(mAP): ')[1].strip())
        info_dict['map'][f'epoch{epoch}'] = map

for ep, losses in info_dict['loss'].items():

    info_dict['loss'][ep] = round(sum(losses) / len(losses), 5)

epoch_losses = list(info_dict['loss'].values())
epoch_maps = list(info_dict['map'].values())[1:]


import matplotlib.pyplot as plt
import numpy as np


epochs = list(range(len(epoch_losses)))  # Assume starting from 0

# Create the figure and axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Left y-axis: Loss
color_loss = 'tab:red'
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', color=color_loss, fontsize=12)
ax1.plot(epochs, epoch_losses, color=color_loss, marker='o', linestyle='-', linewidth=2, markersize=4, label='Loss')
ax1.tick_params(axis='y', labelcolor=color_loss)
ax1.grid(True, alpha=0.3)

# Right y-axis: mAP
ax2 = ax1.twinx()
color_map = 'tab:blue'
ax2.set_ylabel('mAP', color=color_map, fontsize=12)
ax2.plot(epochs, epoch_maps, color=color_map, marker='s', linestyle='-', linewidth=2, markersize=4, label='mAP')
ax2.tick_params(axis='y', labelcolor=color_map)

# add title
plt.title('Training Loss and mAP vs Epoch', fontsize=14, fontweight='bold')

# Add a legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

# Set the x-axis
ax1.set_xlim(min(epochs) - 0.5, max(epochs) + 0.5)
ax1.set_xticks(epochs[::max(1, len(epochs)//10)])  # Automatically adjust the axis ticks

# Adjust the layout
fig.tight_layout()
# show figure
plt.show()

print()