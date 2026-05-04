import json
import warnings
from copy import deepcopy

import torch
import random
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset


def collate_fn(batch):

    return batch

class ADRsDataset(Dataset):

    def __init__(self, data_path, max_labels=256, is_train=True, logger=print, topk_adrs=None, only_single_adrs=False):

        super(ADRsDataset, self).__init__()
        self.logger = logger
        self.data_path = data_path
        self.is_train = is_train
        # self.noise_reactions = [
        #     'off label use', 'product dose omission issue', 'covid-19', 'product use in unapproved indication', 'illness'
        #     'condition aggravated', 'inappropriate schedule of product administration', 'drug dose omission by device',
        #     'incorrect dose administered', 'fall', 'wrong technique in product usage process', 'device difficult to use',
        #     'product storage error', 'product use issue', 'hospitalisation', 'accidental exposure to product',
        #     'therapy interrupted', 'toxicity to various agents', 'infection', 'general physical health deterioration',
        #     'intentional product use issue', 'therapeutic product effect incomplete', 'overdose', 'device issue',
        #     'adverse drug reaction', 'malignant neoplasm progression', 'infusion related reaction', 'treatment failure',
        #     'drug effective for unapproved indication', 'device leakage', 'wrong technique in device usage process',
        #     'adverse event',
        # ]
        self.noise_reactions = [
            'illness', 'condition aggravated', 'fall', 'hospitalisation', 'therapy interrupted',  'adverse event',
            'toxicity to various agents', 'infection', 'general physical health deterioration', 'adverse drug reaction',
            'malignant neoplasm progression', 'infusion related reaction', 'treatment failure',
        ]
        self.noise_reaction_segs = [
            'use', 'error', 'device', 'issue', 'wrong', 'useless', 'incorrect', 'product', 'unapproved', 'technique',
            'inappropriate', 'dose', 'incomplete', 'overdose', 'failure', 'storage', 'without', 'nonspecific',
            'deployment', 'covid-19', 'treatment'
        ]
        self.unii = set()
        self.reactions_hub = set()
        self.reactions_counter = None
        self.reactions_counter_top256 = dict()
        self.max_labels = max_labels
        self.data = self.load_dataset(data_path=self.data_path)
        self.sample_count = len(self.data)

    def _reaction_available_check(self, reaction):

        available = True
        for nrs in self.noise_reaction_segs:
            if nrs in reaction:
                available = False
                break
        if available:
            if reaction in self.noise_reactions:
                available = False

        return available


    def _get_unii_set(self, events):

        uniis = set()
        for event in tqdm(events, desc='getting unii set ...'):
            for drug in event['patient_drug']:
                for unii in drug['unii']:
                    uniis.add(unii)

        import pickle
        with open('unii.pkl', 'wb') as f:
            pickle.dump(uniis, f)

        self.logger(f'unii.pkl save, unii num = {len(uniis)}')

    def load_dataset(self, data_path):

        with open(data_path, 'r') as f:
            data = json.load(f)

        data_ = []
        patient_sex_null_count = 0
        patient_age_null_count = 0
        patient_wei_null_count = 0
        patient_reaction_drug_null_count = 0

        if self.is_train:
            # events = data['events']
            # events = data['events'][0:2048]
            events = data['events'][0:204800]
        else:
            events = data['events'][204800: 204800 + 5000]

        # self._get_unii_set(events=data['events'])

        # construct ADR database
        all_reactions = []
        for item in tqdm(events, desc=f'building reaction hub from {data_path} ...'):
            for pr in item['patient_reaction']:
                # Validate ADR records
                if self._reaction_available_check(pr):
                    self.reactions_hub.add(pr)
                    all_reactions.append(pr)
        assert self.max_labels < len(self.reactions_hub)
        self.reactions_counter = Counter(all_reactions)
        for reaction, count in self.reactions_counter.most_common(256):
            self.reactions_counter_top256[reaction] = count
        reactions_top256 = list(self.reactions_counter_top256.keys())
        reactions_top256_2_id_dict = {reac: i for i, reac in enumerate(reactions_top256)}

        #  data cleaning
        for item in tqdm(events, desc=f'building dataset from {data_path} ...'):

            assert item['patient_sex'] in [0, 1, 2, 'null']
            if item['patient_sex'] == 0 or item['patient_sex'] == 'null':
                item['patient_sex'] = '[UNK]'
                patient_sex_null_count += 1
            elif item['patient_sex'] == 1:
                item['patient_sex'] = 'man'
            elif item['patient_sex'] == 2:
                item['patient_sex'] = 'woman'

            if item['patient_age'] == 'null':
                item['patient_age'] = -1
                patient_age_null_count += 1

            if item['patient_weight'] == 'null':
                item['patient_weight'] = -1
                patient_wei_null_count += 1

            if len(item['patient_reaction']) == 0 or len(item['patient_drug']) == 0:
                patient_reaction_drug_null_count += 1
                continue
            # filter ADR noise
            patient_reaction = []
            for pr in item['patient_reaction']:
                # Validate ADR records
                if self._reaction_available_check(pr):
                    if self.is_train:
                        patient_reaction.append(pr)
                    else:
                        # validation set evaluates only common ADRs
                        if pr in self.reactions_counter_top256:
                            patient_reaction.append(pr)

            if len(patient_reaction) > 0:
                item['patient_reaction'] = patient_reaction
            else:
                # discard the sample
                continue

            #Check whether the UNII is missing; if so, replace it with the [UNK] token in BERT; otherwise, apply whitespace normalization
            for i, drug in enumerate(item['patient_drug']):
                if len(drug['unii']) == 0 or drug['unii'][0] == 'null':
                    item['patient_drug'][i]['unii'] = ['[UNK]']
                else:
                    unii_ = []
                    for ui in drug['unii']:
                        self.unii.add(ui)
                        tmp_str = ''
                        for char in ui:
                            tmp_str = tmp_str + char + ' '
                        unii_.append(tmp_str[:-1])
                    item['patient_drug'][i]['unii'] = unii_

            if self.is_train:
                item['patient_reaction_pos_ids'] = torch.tensor([i for i in range(len(item['patient_reaction']))])
                # Add negative samples
                patient_reaction_neg = random.sample(
                    self.reactions_hub - set(item['patient_reaction']), self.max_labels - len(item['patient_reaction'])
                )
                item['patient_reaction'] += patient_reaction_neg
                assert len(item['patient_reaction']) == self.max_labels
            else:
                item['patient_reaction_pos_ids'] = torch.tensor(
                    [reactions_top256_2_id_dict[reac] for reac in item['patient_reaction']]
                )
                item['patient_reaction'] = reactions_top256

            data_.append(item)

        self.logger(
            f'\ntotal count: {len(events)}, available count: {len(data_)}, '
            f'\npatient sex missing count: {patient_sex_null_count / len(events)}, '
            f'\npatient age missing count: {patient_age_null_count / len(events)}, '
            f'\npatient weight missing count: {patient_wei_null_count / len(events)}, '
            f'\nreaction or drug miss count: {patient_reaction_drug_null_count  / len(events)}, '
            f'\nreaction category num：{len(self.reactions_hub)}, '
            f'\nunii num：{len(self.unii)}, '
        )

        return data_

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        return item


class ADRsDatasetSgTp(Dataset):

    def __init__(self, data_path, is_train=True, logger=print, topk=32, args=None):

        super(ADRsDatasetSgTp, self).__init__()
        self.logger = logger
        self.data_path = data_path
        self.is_train = is_train
        self.shuffle = args.shuffle
        self.use_unii_desc = args.use_unii_desc
        self.unii_desc_file = None
        if args.use_unii_desc:
            self.unii_desc_file = args.unii_desc_file

        self.use_drug_name = args.use_drug_name
        self.noise_reactions = [
            'illness', 'condition aggravated', 'fall', 'hospitalisation', 'therapy interrupted',  'adverse event',
            'toxicity to various agents', 'infection', 'general physical health deterioration', 'adverse drug reaction',
            'malignant neoplasm progression', 'infusion related reaction', 'treatment failure',
            # 'death',
            # 'dermatitis atopic',
            # 'surgery',
        ]
        self.noise_reaction_segs = [
            'use', 'error', 'device', 'issue', 'wrong', 'useless', 'incorrect', 'product', 'unapproved', 'technique',
            'inappropriate', 'dose', 'incomplete', 'overdose', 'failure', 'storage', 'without', 'nonspecific',
            'deployment', 'covid-19', 'treatment', 'ineffective',
        ]
        self.unii = set()
        self.reactions_hub = set()
        self.reactions_counter_topk = dict()
        self.reactions_topk_2_id_dict = dict()
        self.topk = topk
        self.train_rate = 0.8

        self.data = self.load_dataset(data_path=self.data_path)
        self.sample_count = len(self.data)
        self.class_weights = self._get_class_weights()

    def _get_class_weights(self):

        class_weights = []
        num_sample = len(self.data)
        num_class = len(self.reactions_counter_topk)
        for r, c in self.reactions_counter_topk.items():
            weight = max(num_sample / (num_class * c) if c > 0 else 0.0, 0.1)
            class_weights.append(round(weight, 2))
        self.logger('class_weights = {}'.format(class_weights))

        return  torch.tensor(class_weights)

    def _reaction_available_check(self, reaction):

        available = True
        for nrs in self.noise_reaction_segs:
            if nrs in reaction:
                available = False
                break
        if available:
            if reaction in self.noise_reactions:
                available = False

        return available


    def _get_unii_set(self, events):

        uniis = set()
        for event in tqdm(events, desc='getting unii set ...'):
            for drug in event['patient_drug']:
                for unii in drug['unii']:
                    uniis.add(unii)

        import pickle
        with open('unii.pkl', 'wb') as f:
            pickle.dump(uniis, f)


        self.logger(f'unii.pkl save, unii num = {len(uniis)}')

    def _get_reactions_hub_and_topk_reactions(self, events):

        # construct ADR database
        all_reactions = []
        for item in tqdm(events, desc=f'building reaction hub ...'):

            if len(item['patient_reaction']) > 1:
                continue

            for pr in item['patient_reaction']:
                # Validate ADR records
                if self._reaction_available_check(pr):
                    self.reactions_hub.add(pr)
                    all_reactions.append(pr)

        if self.topk > 0:
            reactions_counter_topk = Counter(all_reactions).most_common(self.topk)
        else:
            counter = Counter(all_reactions)
            reactions_counter_topk = counter.most_common(len(counter))

        for reaction, count in reactions_counter_topk:
            self.reactions_counter_topk[reaction] = count

        reactions_topk = list(self.reactions_counter_topk.keys())
        self.reactions_topk_2_id_dict = {reac: i for i, reac in enumerate(reactions_topk)}
        self.logger(f'top {self.topk} reactions = \n{json.dumps(self.reactions_counter_topk, indent=4)}')


    def load_dataset(self, data_path):

        with open(data_path, 'r') as f:
            data = json.load(f)

        if self.use_unii_desc:
            assert self.unii_desc_file is not None
            with open(self.unii_desc_file, 'r') as f:
                self.unii_desc = json.load(f)

        # Truncate the data
        data['events'] = data['events'][0:]
        # data['events'] = data['events'][0:100000]

        patient_sex_null_count = 0
        patient_age_null_count = 0
        patient_wei_null_count = 0
        patient_reaction_drug_null_count = 0

        if self.shuffle:
            random.shuffle(data['events'])

        # clean the data
        events_ = []
        for item in tqdm(data['events'], desc=f'clearing dataset ...'):

            if len(item['patient_reaction']) != 1:
                continue
            # Remove events without ADRs or drug information
            if len(item['patient_reaction']) == 0 or len(item['patient_drug']) == 0:
                patient_reaction_drug_null_count += 1
                continue
            # Filter out noisy ADRs and remove samples with zero valid ADRs
            patient_reaction = []
            for pr in item['patient_reaction']:
                # Validate ADR records
                if self._reaction_available_check(pr):
                    patient_reaction.append(pr)
            if len(patient_reaction) == 0:
                continue
            item['patient_reaction'] = patient_reaction
            # Remove samples with missing UNII codes
            patient_drug = []
            for i, drug in enumerate(item['patient_drug']):
                if len(drug['unii']) == 0 or drug['unii'][0] == 'null':
                    continue
                else:

                    # Normalize special characters in drug active ingredients
                    if '\\' in drug['activesubstancename']:
                        item['patient_drug'][i]['activesubstancename'] = drug['activesubstancename'].replace('\\', ', ')

                    unii_ = []
                    unii_desc = []
                    for ui in drug['unii']:
                        self.unii.add(ui)
                        tmp_str = ''
                        for char in ui:
                            tmp_str = tmp_str + char + ' '
                        unii_.append(tmp_str[:-1])

                        ui_upper = ui.upper()
                        if self.use_unii_desc and ui_upper in self.unii_desc:
                            if 'DrugBank' in self.unii_desc[ui_upper]:
                                unii_desc.append(self.unii_desc[ui_upper]['DrugBank'])

                    item['patient_drug'][i]['unii'] = unii_
                    item['patient_drug'][i]['unii_desc'] = unii_desc
                    if self.use_drug_name and drug['medicinalproduct'] != 'null':
                        item['patient_drug'][i]['unii'].append(drug['medicinalproduct'])
                    if self.use_drug_name and drug['activesubstancename'] != 'null':
                        item['patient_drug'][i]['unii'].append(drug['activesubstancename'])
                    patient_drug.append(item['patient_drug'][i])
            # Determine whether valid drug exposure is present
            if len(patient_drug) == 0:
                continue
            item['patient_drug'] = patient_drug

            # Correct patient demographic information
            # sex
            assert item['patient_sex'] in [0, 1, 2, 'null']
            if item['patient_sex'] == 0 or item['patient_sex'] == 'null':
                item['patient_sex'] = '[UNK]'
                patient_sex_null_count += 1
            elif item['patient_sex'] == 1:
                item['patient_sex'] = 'man'
            elif item['patient_sex'] == 2:
                item['patient_sex'] = 'woman'

            # weight
            item['patient_weight_'] = item['patient_weight']
            if item['patient_weight'] == 'null' or item['patient_age'] == 'null' or item['patient_weight'] <= 0 or item['patient_age'] <= 0:
                item['patient_weight'] = '[UNK]'
                patient_wei_null_count += 1
            elif item['patient_age'] <=12:
                if item['patient_weight'] < (0.8* (item['patient_age'] * 2 + 8)):
                    item['patient_weight'] = 'thin'
                elif item['patient_weight'] > (1.2* (item['patient_age'] * 2 + 8)):
                    item['patient_weight'] = 'obesity'
                else:
                    item['patient_weight'] = 'normal'
            elif item['patient_age'] <=18:
                if item['patient_weight'] < (0.8* (item['patient_age'] * 3 - 2)):
                    item['patient_weight'] = 'thin'
                elif item['patient_weight'] > (1.2* (item['patient_age'] * 3 - 2)):
                    item['patient_weight'] = 'obesity'
                else:
                    item['patient_weight'] = 'normal'
            else:
                if item['patient_sex'] == 'man':
                    if item['patient_weight'] < 50:
                        item['patient_weight'] = 'thin'
                    elif item['patient_weight'] > 75:
                        item['patient_weight'] = 'obesity'
                    else:
                        item['patient_weight'] = 'normal'
                else:
                    if item['patient_weight'] < 40:
                        item['patient_weight'] = 'thin'
                    elif item['patient_weight'] > 65:
                        item['patient_weight'] = 'obesity'
                    else:
                        item['patient_weight'] = 'normal'

            # age
            item['patient_age_'] = item['patient_age']  # copy
            if item['patient_age'] == 'null':
                item['patient_age'] = '[UNK]'
                patient_age_null_count += 1
            elif item['patient_age'] < 12:
                item['patient_age'] = 'child'
            elif item['patient_age'] < 18:
                item['patient_age'] = 'teenager'
            elif item['patient_age'] < 50:
                item['patient_age'] = 'adults'
            else:
                item['patient_age'] = 'elder'

            events_.append(item)

        self.logger(
            f"raw events num = {len(data['events'])}, clear events num = {len(events_)}"
            f"\npatient sex missing count: {patient_sex_null_count / len(data['events'])}, "
            f"\npatient age missing count: {patient_age_null_count / len(data['events'])}, "
            f"\npatient weight missing count: {patient_wei_null_count / len(data['events'])}, "
            f"\nreaction or drug miss count: {patient_reaction_drug_null_count  / len(data['events'])}, "
            f"\nreaction category num：{len(self.reactions_hub)}, "
            f"\nunii num：{len(self.unii)}, "
        )

        data['events'] = events_

        self._get_reactions_hub_and_topk_reactions(events=data['events'])

        # data splitting
        if self.is_train:
            dataset_type = 'Train'
            events = data['events'][0:int(len(data['events']) * self.train_rate)]
        else:
            dataset_type = 'Test'
            events = data['events'][int(len(data['events']) * self.train_rate): ]

        # Retain only samples corresponding to the top-k ADRs
        events_ = []
        single_drug_num = 0
        multi_drug_num = 0
        for item in tqdm(events, desc=f'building {dataset_type} dataset ...'):
            patient_reaction = [pr for pr in item['patient_reaction'] if pr in self.reactions_counter_topk]
            if len(patient_reaction) == 0:
                continue
            # Update ADR annotations
            item['patient_reaction_'] = item['patient_reaction']
            item['patient_reaction'] = patient_reaction
            # Assign positive ADR IDs
            item['patient_reaction_pos_ids'] = torch.tensor(
                [self.reactions_topk_2_id_dict[reac] for reac in item['patient_reaction']]
            )
            # Construct lists of positive and negative ADRs
            item['patient_reaction'] = list(self.reactions_topk_2_id_dict.keys())

            if len(item['patient_drug']) > 1:
                multi_drug_num += 1
            else:
                single_drug_num += 1

            events_.append(item)

        self.logger(
            f"all the single event num = {len(events)}, available {dataset_type}, events num = {len(events_)}, "
            f"single drug num = {single_drug_num}, multi drug num = {multi_drug_num}"
        )

        if not self.is_train:

            # Compute the occurrence and frequency of drug combinations
            unit_drugs_set = set()
            unit_drugs_list = []
            unit_drugs_man_list = []
            unit_drugs_woman_list = []
            for event in events_:
                unit_drug_set = set()
                for patient_drug in event['patient_drug']:
                    unit_drug_set.add(patient_drug['medicinalproduct'].lower())
                unit_drug_list = list(unit_drug_set)
                unit_drug_list.sort()
                unit_drug = ''
                for it in unit_drug_list:
                    unit_drug = unit_drug + it + '_'
                unit_drugs_set.add(unit_drug[:-1])
                unit_drugs_list.append(unit_drug[:-1])
                if event['patient_sex'] == 'man':
                    unit_drugs_man_list.append(unit_drug[:-1])
                elif event['patient_sex'] == 'woman':
                    unit_drugs_woman_list.append(unit_drug[:-1])

            unit_drugs_counter = Counter(unit_drugs_list)
            unit_drugs_man_counter = Counter(unit_drugs_man_list)
            unit_drugs_woman_counter = Counter(unit_drugs_woman_list)

            unit_drugs_counter_topk = unit_drugs_counter.most_common(32)
            unit_drugs_man_counter_topk = unit_drugs_man_counter.most_common(32)
            unit_drugs_woman_counter_topk = unit_drugs_woman_counter.most_common(32)

        return events_

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        return item


class ADRsDatasetSgTpAndMultiDrug(Dataset):

    def __init__(self, dataset=None, logger=print):
        super(ADRsDatasetSgTpAndMultiDrug, self).__init__()
        self.logger = logger
        self.original_data = deepcopy(dataset.data)
        self.valid_indices = self._get_valid_indices(self.original_data)

    def _get_valid_indices(self, dataset):

        valid_indices = [
            i for i, item in enumerate(dataset) if len(item['patient_drug']) > 1
        ]

        self.logger(f'multi drug dataset, sample num = {len(valid_indices)}')

        return valid_indices

    def __len__(self):

        return len(self.valid_indices)

    def __getitem__(self, idx):

        org_idx = self.valid_indices[idx]
        item = self.original_data[org_idx]

        return item




if __name__ == '__main__':

    from torch.utils.data import DataLoader, DistributedSampler
    from pretrain import get_args_parser
    args = get_args_parser()

    data_path = '../output/adverse_event_2024.json'
    dataset_train = ADRsDatasetSgTp(data_path, is_train=True, args=args)
    dataset_valid = ADRsDatasetSgTp(data_path, is_train=False, args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, batch_size=args.batch_size, drop_last=True)
    data_loader_train = DataLoader(
        dataset=dataset_train, batch_sampler=batch_sampler_train, collate_fn=collate_fn, num_workers=args.num_workers
    )

    for step, batch_input in enumerate(data_loader_train):
        print()

    # dataset = ADRsDataset(data_path)
    # data_loader = torch.utils.data.DataLoader(
    #     dataset=dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn
    # )
    # for batch_input in data_loader:
    #     print()
    # print()



