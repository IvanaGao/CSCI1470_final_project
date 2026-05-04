import json
import os

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


def get_data_statistics_info(data_root_dir, years):

    for year in years:
        json_files = [
            # file for file in os.listdir(os.path.join(data_root_dir, year)) if '.json' in file and '.json.zip' not in file
            file for file in os.listdir(os.path.join(data_root_dir, year)) if '.json' in file and '.json.zip' not in file
        ]

    data_info = {
        'patient.patientonsetage': {'value': [], 'null_num': 0,},
        'patient.patientsex': {'value': [], 'null_num': 0,},
        'patient.patientweight': {'value': [], 'null_num': 0,},
        'patient.reaction.reactionmeddrapt': {'value': [], 'null_num': 0, 'counter': None},
        'serious': {'value': [], 'null_num': 0,},
        'patient.num': 0,
    }

    null_count_patientonsetage = 0
    null_count_patientsex = 0
    null_count_patientweight = 0
    null_count_reactionmeddrapt = 0
    null_count_serious = 0
    for i, json_file in enumerate(json_files):

        # file_path = os.path.join(data_root_dir, year, json_file)
        file_path = data_root_dir + year + '/' + json_file
        print('\n[{}/{}] analysing {}'.format(i + 1, len(json_files), file_path))
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data_info['patient.num'] += len(data['results'])

        for item in tqdm(data['results'], desc='patient.patientonsetage analysis ...'):

            if 'patient' in item:
                if 'patientonsetage' in item['patient']:
                    try:
                        age = int(item['patient']['patientonsetage'])
                        if age > 100:
                            null_count_patientonsetage += 1
                        else:
                            data_info['patient.patientonsetage']['value'].append(age)
                    except:
                        null_count_patientonsetage += 1
                else:
                    null_count_patientonsetage += 1
        data_info['patient.patientonsetage']['null_num'] = null_count_patientonsetage

        for item in tqdm(data['results'], desc='patient.patientsex analysis ...'):

            if 'patient' in item:
                if 'patientsex' in item['patient']:
                    try:
                        sex = int(item['patient']['patientsex'])
                        data_info['patient.patientsex']['value'].append(sex)
                    except:
                        null_count_patientsex += 1
                else:
                    null_count_patientsex += 1
        data_info['patient.patientsex']['null_num'] = null_count_patientonsetage


        for item in tqdm(data['results'], desc='patient.patientweight analysis ...'):

            if 'patient' in item:
                if 'patientweight' in item['patient']:
                    try:
                        weight = int(float(item['patient']['patientweight']))
                        if weight > 250:
                            null_count_patientweight += 1
                        else:
                            data_info['patient.patientweight']['value'].append(weight)
                    except:
                        null_count_patientweight += 1
                else:
                    null_count_patientweight += 1
        data_info['patient.patientweight']['null_num'] = null_count_patientonsetage

        for item in tqdm(data['results'], desc='patient.reaction.reactionmeddrapt analysis ...'):

            if 'patient' in item:
                if 'reaction' in item['patient']:
                    for reaction in item['patient']['reaction']:
                        if 'reactionmeddrapt' in reaction and len(reaction['reactionmeddrapt']) > 0:
                            data_info['patient.reaction.reactionmeddrapt']['value'].append(
                                reaction['reactionmeddrapt'].lower()
                            )
                            print()
                        else:
                            null_count_reactionmeddrapt += 1
                else:
                    null_count_reactionmeddrapt += 1
        data_info['patient.reaction.reactionmeddrapt']['null_num'] = null_count_patientonsetage

        for item in tqdm(data['results'], desc='serious analysis ...'):

            if 'serious' in item:
                try:
                    age = int(item['serious'])
                    data_info['serious']['value'].append(age)
                except:
                    null_count_serious += 1
        data_info['serious']['null_num'] = null_count_patientonsetage

        # break

    return data_info



if __name__ == '__main__':

    years = ['2024']
    data_info = get_data_statistics_info(
        # data_root_dir='./dataset/adverse_event_data/',
        data_root_dir='/workspace/mount/b100_zaip_data/xxx/datasets/adrs/adverse_event_data/',
        years=years
    )

    # build a histogram
    t = data_info['patient.patientonsetage']['value']
    plt_pdf(
        data=data_info['patient.patientonsetage']['value'], bin=100,
        desc='patient.patientonsetage PDF\n(count = {}, miss_rate = {})'.format(
            data_info['patient.num'], round(data_info['patient.patientonsetage']['null_num'] / data_info['patient.num'], 2)
        )
    )

    # build a histogram
    plt_pdf(
        data=data_info['patient.patientsex']['value'], bin=100,
        desc='patient.patientsex (1-man 2-woman) PDF\n(count = {}, miss_rate = {})'.format(
            data_info['patient.num'], round(data_info['patient.patientsex']['null_num'] / data_info['patient.num'], 2)
        )
    )

    # build a histogram
    plt_pdf(
        data=data_info['patient.patientweight']['value'], bin=100,
        desc='patient.patientweight (kg) PDF\n(count = {}, miss_rate = {})'.format(
            data_info['patient.num'], round(data_info['patient.patientweight']['null_num'] / data_info['patient.num'], 2)
        )
    )

    count = Counter(data_info['patient.reaction.reactionmeddrapt']['value'])
    data_info['patient.reaction.reactionmeddrapt']['counter'] = count
    top1000_str_counts = count.most_common(1000)
    from collections import OrderedDict
    top_event_count = OrderedDict()
    for (e, c) in top1000_str_counts:
        top_event_count[e] = c
        # print(e, c)
    file_name = ''
    for year in years:
        file_name += year + '_'
    file_name = file_name[:-1]
    with open('./output/top_event_count_{}.json'.format(file_name), 'w') as f:
        json.dump(top_event_count, f, indent=4)

    # build a histogram
    plt_pdf(
        data=data_info['serious']['value'], bin=100,
        desc='serious PDF\n(count = {}, miss_rate = {})'.format(
            data_info['patient.num'], round(data_info['serious']['null_num'] / data_info['patient.num'], 2)
        )
    )

    with open('./output/statistics_{}.json'.format(file_name), 'w') as f:
        json.dump(data_info, f)

    print()


