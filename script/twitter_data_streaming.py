import time
import json
import requests
from getpass import getpass

bearer_token = getpass("Enter bearer token: ")

headers = {
    'Authorization': "Bearer {}".format(bearer_token),
    'User-Agent': 'v2SampledStreamPython'
    }

url = "https://api.twitter.com/2/tweets/search/stream/rules"

# function to get existing rules for filtering Twitter stream
def get_rules():
    response = requests.get(url, headers = headers)
    if response.status_code != 200:
        raise Exception("Cannot get rules (HTTP {}): {}".format(response.status_code, response.text))
        return response.json()

# function to set new rules for filtering Twitter stream
def set_rules(sample_rules):
    payload = {'add': sample_rules}
    response = requests.post(url, headers = headers, json = payload)
    if response.status_code != 201:
        raise Exception("Cannot add rules (HTTP {}): {}".format(response.status_code, response.text))
    else:
        print("Successfully set the rules!")

# function to remove all current rules for filtering Twitter stream
def delete_all_rules(current_rules):
    if current_rules is None or 'data' not in current_rules:
        return None
    
    ids = []
    for rule in current_rules['data']:
        ids.append(rule['id'])
        
    payload = {'delete': {'ids': ids}}
    response = requests.post(url, headers = headers, json = payload)
    
    if response.status_code != 200:
        raise Exception("Cannot delete rules (HTTP {}): {}".format(response.status_code, response.text))
    else:
        print("Successfully deleted the rules!")
        
current_rules = get_rules()
print(current_rules)

# building rules
# https://developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/integrate/build-a-rule

sample_rules = [
    {'value': '(crypto OR cryptocurrency) lang:en -is:retweet', 'tag': 'crypto'},
    {'value': 'bitcoin lang:en -is:retweet', 'tag': 'bitcoin'},
    {'value': 'dogecoin lang:en -is:retweet', 'tag': 'dogecoin'},
    {'value': 'Musk lang:en -is:retweet', 'tag': 'musk'},
    ]

set_rules(sample_rules)

current_rules = get_rules()
print(current_rules)

