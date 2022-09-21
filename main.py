import re


def take_first(elem):
    return elem[0]


raw_data={}

len = int(input(""))
for i in range(len):
    tmp_str = input(f"")
    tmp_list = re.split(',', tmp_str)
    up, down = [int(j) for j in tmp_list]
    if up in raw_data.keys():
        raw_data[up]+=1
    else:
        raw_data.update({up:1})

    if down in raw_data.keys():
        raw_data[down]-=1
    else:
        raw_data.update({down:-1})

data_list = list(raw_data.items())
data_list.sort(key=take_first)
rtn = 0
last_key = 0
level = 0
for key,change in data_list:
    if level == 1:
        rtn += key-last_key
    level += change
    last_key = key

print(rtn)
