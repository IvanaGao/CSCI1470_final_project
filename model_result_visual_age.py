import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter

result_file = './log/0812-6/evaluation_result_all9.pkl'
# result_file = './log/0812-6/evaluation_result_mtd9.pkl'

with open(result_file, 'rb') as f:
    events = pickle.load(f)

# Analyze polypharmacy patterns and their frequencies
unit_drugs_set = set()
unit_drugs_list = []
unit_drugs_child_teenager_list = []
unit_drugs_adults_list = []
unit_drugs_elder_list = []
for event in tqdm(events, desc='age ...'):
    # Avoid the impact of duplicate drug entries
    unit_drug_set = set()
    for patient_drug in event['patient_drug']:
        assert '_' not in patient_drug['medicinalproduct']
        unit_drug_set.add(patient_drug['medicinalproduct'])

    unit_drug_list = list(unit_drug_set)
    unit_drug_list.sort()   # Avoid the impact of order variation
    unit_drug = ''
    for it in unit_drug_list:
        unit_drug = unit_drug + it + '_'
    unit_drugs_set.add(unit_drug[:-1])
    unit_drugs_list.append(unit_drug[:-1])
    event['multi_unit_drugs'] = unit_drug[:-1]

    if event['patient_age'] in ['child', 'teenager']:
        unit_drugs_child_teenager_list.append(unit_drug[:-1])
    elif event['patient_age'] == 'adults':
        unit_drugs_adults_list.append(unit_drug[:-1])
    elif event['patient_age'] == 'elder':
        unit_drugs_elder_list.append(unit_drug[:-1])

# Polypharmacy
multi_unit_drugs_list = [item for item in unit_drugs_list if '_' in item]
multi_unit_drugs_child_teenager_list = [item for item in unit_drugs_child_teenager_list if '_' in item]
multi_unit_drugs_adults_list = [item for item in unit_drugs_adults_list if '_' in item]
multi_unit_drugs_elder_list = [item for item in unit_drugs_elder_list if '_' in item]

multi_unit_drugs_counter = Counter(multi_unit_drugs_list)

multi_unit_drugs_counter_topk = multi_unit_drugs_counter.most_common()

# Compute the intersection of polypharmacy sets across genders
common_multi_drugs_set1 = set(multi_unit_drugs_child_teenager_list) & set(multi_unit_drugs_adults_list)
common_multi_drugs_set2 = set(multi_unit_drugs_child_teenager_list) & set(multi_unit_drugs_elder_list)
common_multi_drugs_set3 = set(multi_unit_drugs_adults_list) & set(multi_unit_drugs_elder_list)

common_multi_drugs_set = set(multi_unit_drugs_child_teenager_list) & set(multi_unit_drugs_adults_list) & set(multi_unit_drugs_elder_list)

common_multi_drugs_list = []
for mtd, num in multi_unit_drugs_counter_topk:
    if mtd in common_multi_drugs_set:
        common_multi_drugs_list.append(mtd)
        print(mtd, '\t', num)


# visualization
for comm_multi_drug in common_multi_drugs_list:

    teenager_reaction_pros = []
    adults_reaction_pros = []
    elder_reaction_pros = []
    for event in events:
        if event['multi_unit_drugs'] == comm_multi_drug:

            if event['patient_age'] in ['child', 'teenager']:
                teenager_reaction_pros.append(event['pred'])
            elif event['patient_age'] == 'adults':
                adults_reaction_pros.append(event['pred'])
            elif event['patient_age'] == 'elder':
                elder_reaction_pros.append(event['pred'])

    teenager_reaction_pro_mean = torch.stack(teenager_reaction_pros, dim=0).mean(dim=0)
    adults_reaction_pro_mean = torch.stack(adults_reaction_pros, dim=0).mean(dim=0)
    elder_reaction_pro_mean = torch.stack(elder_reaction_pros, dim=0).mean(dim=0)

    KL_m2w = -1
    # KL_m2w = F.kl_div(
    #     input=F.log_softmax(man_reaction_pro_mean, dim=0),
    #     target=F.softmax(woman_reaction_pro_mean, dim=0),
    #     reduction='batchmean',
    #     log_target=False
    # )
    # # KL_w2m = F.kl_div(input=woman_reaction_pro_mean, target=man_reaction_pro_mean, reduction='batchmean', log_target=False)
    # KL_m2w = round(KL_m2w.item(), 5)
    # # KL_w2m = round(KL_w2m.item(), 5)

    # Create a figure canvas
    plt.figure(figsize=(12, 4))
    x = np.arange(len(teenager_reaction_pro_mean))

    # Plot the distribution curve
    plt.plot(x, teenager_reaction_pro_mean, 'b--', linewidth=2, label='teenager', alpha=0.7)
    plt.plot(x, adults_reaction_pro_mean, 'r--', linewidth=2, label='adults', alpha=0.7)
    plt.plot(x, elder_reaction_pro_mean, 'g--', linewidth=2, label='elder', alpha=0.7)
    # Fill the area
    plt.fill_between(x, teenager_reaction_pro_mean, color='blue', alpha=0.1)
    plt.fill_between(x, adults_reaction_pro_mean, color='red', alpha=0.1)
    plt.fill_between(x, elder_reaction_pro_mean, color='green', alpha=0.1)

    # Add annotations
    plt.title(f'Probability Distributions for {comm_multi_drug} '
              f'te={len(teenager_reaction_pros)}, '
              f'ad={len(adults_reaction_pros)}'
              f'ed={len(elder_reaction_pros)}'
              f'\nKL_m2w = {KL_m2w}')
    plt.xlabel('Reaction ID')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # show figure
    plt.tight_layout()
    plt.show()
    print()


print()