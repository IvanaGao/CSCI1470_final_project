import os
import shutil
import requests
import zipfile
from bs4 import BeautifulSoup


with open('./openFDA.html', 'r', encoding='utf-8') as file:
    html_data = file.read()

soup = BeautifulSoup(html_data, 'html.parser')

links = []
for link in soup.find_all('a'):
    if '.json.zip' in link.get('href') and 'drug-event-' in link.get('href'):
        links.append(link.get('href'))

# Retrieve the current database records list
# data_root_dir = './adverse_event_data/'
data_root_dir = 'F:/ADRs/dataset/adverse_event_data/'
# Scrape data for a specified year
years = ['2023']
for year in years:
    # Create a directory for saving data
    data_root_dir_sub = os.path.join(data_root_dir, year)
    os.makedirs(data_root_dir_sub, exist_ok=True)
    exist_files = os.listdir(data_root_dir_sub)
    sub_links = [lk for lk in links if '/{}'.format(year) in lk]
    for i, link in enumerate(sub_links):
        if year in link:
            file_name = link.split('/')[-2]+ '_' + link.split('/')[-1]
            if file_name not in exist_files:
                try:
                    headers = {'User-Agent': 'Chrome/96.0.4664.45 Safari/537.36'}
                    print('[{}/{}] downloading {} to {}'.format(i+1, len(sub_links), link, data_root_dir_sub))
                    r = requests.get(link, headers=headers, timeout=5)
                    with open(os.path.join(data_root_dir_sub, file_name), 'wb') as f:
                        f.write(r.content)
                    print('download success')
                except Exception as e:
                    print(e)
                    print('download error: ', link)
            else:
                print(file_name +' has already been downloaded')

            # unzip files
            if file_name.replace('.zip', '') not in exist_files:
                with zipfile.ZipFile(os.path.join(data_root_dir_sub, file_name), 'r') as zip_file:
                    for file in zip_file.namelist():
                        if file.endswith('.json'):
                            try:
                                print('unzipping: {}'.format(file_name))
                                zip_file.extract(file, './tmp')
                                print('unzip success')
                                shutil.move('./tmp/{}'.format(file), os.path.join(data_root_dir_sub, file_name.replace('.zip', '')))
                            except Exception as e:
                                print(e)
                                print('unzip error: {}'.format(file_name))
            else:
                print('{} already extracted'.format(file_name.replace('.zip', '')))


print()