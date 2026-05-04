import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter

result_file = './log/0812-7/evaluation_result_all9.pkl'
# result_file = './log/0812-7/evaluation_result_mtd9.pkl'

with open(result_file, 'rb') as f:
    events = pickle.load(f)

# Analyze polypharmacy patterns and their frequencies
unit_drugs_set = set()
unit_drugs_list = []
unit_drugs_man_list = []
unit_drugs_woman_list = []
for event in tqdm(events, desc='gender ...'):
    # Avoid the impact of duplicate drug entries
    unit_drug_set = set()
    for patient_drug in event['patient_drug']:
        assert '_' not in patient_drug['medicinalproduct']
        unit_drug_set.add(patient_drug['medicinalproduct'])

    unit_drug_list = list(unit_drug_set)
    unit_drug_list.sort()   #Avoid the impact of order variation
    unit_drug = ''
    for it in unit_drug_list:
        unit_drug = unit_drug + it + '_'
    unit_drugs_set.add(unit_drug[:-1])
    unit_drugs_list.append(unit_drug[:-1])
    event['multi_unit_drugs'] = unit_drug[:-1]

    if event['patient_sex'] == 'man':
        unit_drugs_man_list.append(unit_drug[:-1])
    elif event['patient_sex'] == 'woman':
        unit_drugs_woman_list.append(unit_drug[:-1])


# Polypharmacy
multi_unit_drugs_list = [item for item in unit_drugs_list if '_' in item]
multi_unit_drugs_man_list = [item for item in unit_drugs_man_list if '_' in item]
multi_unit_drugs_woman_list = [item for item in unit_drugs_woman_list if '_' in item]

multi_unit_drugs_counter = Counter(multi_unit_drugs_list)
multi_unit_drugs_counter_topk = multi_unit_drugs_counter.most_common()

# Compute the intersection of polypharmacy sets across genders
common_multi_drugs_set = set(multi_unit_drugs_man_list) & set(multi_unit_drugs_woman_list)

common_multi_drugs_list = []
for mtd, num in multi_unit_drugs_counter_topk:
    if mtd in common_multi_drugs_set:
        common_multi_drugs_list.append(mtd)
        print(mtd, '\t', num)

# visualization
for comm_multi_drug in common_multi_drugs_list:

    man_reaction_pros = []
    woman_reaction_pros = []
    for event in events:
        if event['multi_unit_drugs'] == comm_multi_drug:
            if event['patient_sex'] == 'man':
                man_reaction_pros.append(event['pred'])
            elif event['patient_sex'] == 'woman':
                woman_reaction_pros.append(event['pred'])
    man_reaction_pro_mean = torch.stack(man_reaction_pros, dim=0).mean(dim=0)
    woman_reaction_pro_mean = torch.stack(woman_reaction_pros, dim=0).mean(dim=0)

    KL_m2w = F.kl_div(
        input=F.log_softmax(man_reaction_pro_mean, dim=0),
        target=F.softmax(woman_reaction_pro_mean, dim=0),
        reduction='batchmean',
        log_target=False
    )
    # KL_w2m = F.kl_div(input=woman_reaction_pro_mean, target=man_reaction_pro_mean, reduction='batchmean', log_target=False)
    KL_m2w = round(KL_m2w.item(), 5)
    # KL_w2m = round(KL_w2m.item(), 5)

    # Create a figure canvas
    plt.figure(figsize=(12, 4))
    x = np.arange(len(man_reaction_pro_mean))

    # Plot the distribution curve
    plt.plot(x, man_reaction_pro_mean, 'b-', linewidth=2, label='man', alpha=0.7)
    plt.plot(x, woman_reaction_pro_mean, 'r--', linewidth=2, label='woman', alpha=0.7)
    # Fill the area
    plt.fill_between(x, man_reaction_pro_mean, color='blue', alpha=0.1)
    plt.fill_between(x, woman_reaction_pro_mean, color='red', alpha=0.1)



    # Add annotations
    plt.title(f'Probability Distributions for {comm_multi_drug} m={len(man_reaction_pros)}, w={len(woman_reaction_pros)}\nKL_m2w = {KL_m2w}')
    plt.xlabel('Reaction ID')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # show figure
    plt.tight_layout()
    plt.show()

    print()


print()