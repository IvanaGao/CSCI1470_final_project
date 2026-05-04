import json
import os
import traceback
from tqdm import tqdm
from collections import Counter
from matplotlib import pyplot as plt


def plt_pdf(data=None, bin=10, desc=''):

    # build a histogram
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=bin, color='skyblue', edgecolor='black')
    # add title and label
    plt.title(desc)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    # show grid
    plt.grid(True)
    plt.tight_layout()
    # show figure
    plt.show()
    plt.close()


def build_dataset(data_root_dir, years):

    for year in years:
        json_files = [
            file for file in os.listdir(os.path.join(data_root_dir, year)) if '.json' in file and '.json.zip' not in file
        ]

    data_info = {
        'year': years,
        'source': 'openFDA',
        'event_categories': [],
        'events': [],
    }

    event_id = 0
    for i, json_file in enumerate(json_files):

        if i > 1:
            break

        # file_path = os.path.join(data_root_dir, year, json_file)
        file_path = data_root_dir + '/' + year + '/' + json_file
        print('\n[{}/{}] analysing {}'.format(i + 1, len(json_files), file_path))
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for event in tqdm(data['results'], desc='event analysis ...'):

            event_item = {
                'event_id': event_id,
                'patient_sex': 'null',
                'patient_age': -1,
                'patient_weight': 'null',
                'patient_reaction': [],
                'patient_drug': [],
                'serious': 'null',
            }

            # Retrieve and validate patient age
            if 'patient' in event:
                if 'patientonsetage' in event['patient']:
                    try:
                        age = int(event['patient']['patientonsetage'])
                        if age < 100:
                            event_item['patient_age'] = age
                    except:
                        print('error: age = {}'.format(event['patient']['patientonsetage']))
            else:
                continue

            # Retrieve and validate patient sex
            if 'patient' in event:
                if 'patientsex' in event['patient']:
                    try:
                        sex = int(event['patient']['patientsex'])
                        event_item['patient_sex'] = sex
                    except:
                        print('error: sex = {}'.format(event['patient']['patientsex']))

            if 'patient' in event:
                if 'patientweight' in event['patient']:
                    try:
                        weight = int(float(event['patient']['patientweight']))
                        if weight < 250:
                            event_item['patient_weight'] = weight
                    except:
                        traceback.print_exc()
                        print('error: weight = {}'.format(event['patient']['patientweight']))

            if 'patient' in event:
                if 'reaction' in event['patient'] and len(event['patient']['reaction']) > 0:
                    for reaction in event['patient']['reaction']:
                        if 'reactionmeddrapt' in reaction and len(reaction['reactionmeddrapt']) > 0:
                            event_item['patient_reaction'].append(reaction['reactionmeddrapt'].lower())
                    event_item['patient_reaction'] = list(set(event_item['patient_reaction']))
                else:
                    continue

            if 'patient' in event:
                if 'drug' in event['patient'] and len(event['patient']['drug']) > 0:
                    drug_date = {}
                    for drug in event['patient']['drug']:

                        medicinalproduct = drug['medicinalproduct'] if 'medicinalproduct' in drug else 'null'
                        activesubstancename = drug['activesubstance']['activesubstancename'] if 'activesubstance' in drug and 'activesubstancename' in drug['activesubstance'] else 'null'
                        unii = drug['openfda']['unii'] if 'openfda' in drug and 'unii' in drug['openfda'] else ['null']

                        if 'drugstartdate' in drug and 'drugenddate' in drug:
                            start_date = drug['drugstartdate']
                            end_date = drug['drugenddate']
                        else:
                            start_date = 'null'
                            end_date = 'null'

                        dg = medicinalproduct + '_' + activesubstancename
                        for un in unii:
                            dg = dg + '_' + un.lower()

                        if dg not in drug_date:
                            drug_date[dg] = (start_date, end_date)
                        elif start_date != 'null':
                            drug_date[dg] = (start_date, end_date)

                    for drug, date in drug_date.items():
                        patient_drug = {}
                        patient_drug['medicinalproduct'] = drug.split('_')[0]
                        patient_drug['activesubstancename'] = drug.split('_')[1]
                        patient_drug['unii'] = drug.split('_')[2:]

                        # if len(patient_drug['unii']) == 0:
                        #     print()

                        patient_drug['start_date'], patient_drug['end_date'] = date
                        event_item['patient_drug'].append(patient_drug)

                        # patient_drug = {
                        #     'medicinalproduct': 'null',
                        #     'activesubstancename': 'null',
                        #     'unii': [],
                        #     # 'start_date': 'null',
                        #     # 'end_date': 'null',
                        # }
                        # if 'medicinalproduct' in drug:
                        #     patient_drug['medicinalproduct'] = drug['medicinalproduct']
                        # if 'activesubstance' in drug and 'activesubstancename' in drug['activesubstance']:
                        #     patient_drug['activesubstancename'] = drug['activesubstance']['activesubstancename']
                        # if 'openfda' in drug and 'unii' in drug['openfda']:
                        #     patient_drug['unii'] = drug['openfda']['unii']
                        # # if 'drugstartdate' in drug and len(drug['drugstartdate']) > 0:
                        # #     patient_drug['start_date'] = drug['drugstartdate']
                        # # if 'drugenddate' in drug and len(drug['drugenddate']) > 0:
                        # #     patient_drug['end_date'] = drug['drugenddate']
                        #
                        # event_item['patient_drug'].append(patient_drug)

                else:
                    continue

            if 'serious' in event:
                try:
                    serious = int(event['serious'])
                    event_item['serious'] = serious
                except:
                    print('error: serious = {}'.format(event['serious']))

            data_info['events'].append(event_item)
            event_id += 1
        # print()

    return data_info


if __name__ == '__main__':

    years = ['2024']
    data_info = build_dataset(
        data_root_dir='/workspace/mount/b100_zaip_data/xxx/datasets/adrs/adverse_event_data/',
        years=years
    )

    file_name = ''
    for year in years:
        file_name += year + '_'
    file_name = file_name[:-1]

    with open('./output/adverse_event_{}.json'.format(file_name), 'w') as f:
        json.dump(data_info, f)

    print()


