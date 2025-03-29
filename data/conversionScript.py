import json

# Read the JSON file

with open('data/restaurants.json','r') as f:
    data = json.load(f)

for restaurant in data:
    restaurant['tags'].append(restaurant['price'])
    del restaurant['price']

# Write back to file
with open('data/restaurants.json', 'w') as f:
    json.dump(data, f, indent=2)
