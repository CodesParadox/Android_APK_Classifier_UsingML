import pandas as pd
import json
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def is_2d(value):
    if isinstance(value, list) and all(isinstance(i, list) for i in value):
        return True
    return False


def count_apps(app_list):
    malicious_count = 0
    benign_count = 0
    for item in app_list:
        if int(item['label']) == 1:
            malicious_count += 1
        else:
            benign_count += 1
    return malicious_count, benign_count


def get_random_number(start, end, isInt=True):
    rand = random.randrange(start, end+1)
    randU = random.uniform(start, end)
    if isInt:
        while rand == 0:
            rand = random.randrange(start, end+1)
        return rand
    while randU == 0:
        randU = random.uniform(start, end)
    return randU


def filter_apps(num_malicious, num_benign, app_list):
    malicious_apps = [app for app in app_list if app["label"] == 1]
    benign_apps = [app for app in app_list if app["label"] == 0]

    filtered_malicious_apps = random.sample(malicious_apps, num_malicious)
    filtered_benign_apps = random.sample(benign_apps, num_benign)

    filtered_apps = filtered_malicious_apps + filtered_benign_apps
    random.shuffle(filtered_apps)
    return filtered_apps


def removePropertyFromJson(property, json):
    for i in range(len(json)):
        del json[i][property]

    return json


def load_data(filename):
    """
    This function loads data from a JSON file and returns a Pandas DataFrame. If the DataFrame contains any NaN values, they are replaced with 0.

    Inputs:
        filename (str): The filepath of the JSON file to be loaded.

    Returns:
        result (pandas.DataFrame): The resulting DataFrame containing the data from the JSON file.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    data = removePropertyFromJson('sha256', data)

    malicious_count, benign_count = count_apps(data)
    print(f'malicious_apps_size in json file = {malicious_count}')
    print(f'benign_apps_size in json file = {benign_count}')

    benign_count = int(benign_count * 0.9)
    malicious_count = int(benign_count * 0.1)
    data = filter_apps(benign_count, malicious_count, data)
    print(f'malicious_apps_size selected by 0.1 = {malicious_count}')
    print(f'benign_apps_size selected by 0.9 = {benign_count}')
    print(f'total application = {len(data)}')
    print()
    malicious_count, benign_count = count_apps(data)
    result = pd.DataFrame(data)
    result = result.fillna(0)
    print('result', result)
    return result, malicious_count, benign_count
