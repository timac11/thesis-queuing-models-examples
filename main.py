import requests

resp = requests.get('http://localhost:5555/api/tasks?state=SUCCESS')

res = resp.json()

for _, value in res.items():
    print(value['runtime'])
    print(value['result'])
    print(value['name'])
    print(value['result'])

