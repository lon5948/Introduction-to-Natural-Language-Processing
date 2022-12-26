import json

D = [[], [], []]

with open("data/train_HW3dataset.json", "r", encoding="utf8") as f:
    data = json.load(f)
    for i in range(len(data)):
        for j in range(len(data[i][1])):
            d = ['\n'.join(data[i][0]).lower(), data[i][1][j]["question"].lower()]
            for k in range(len(data[i][1][j]["choice"])):
                d += [data[i][1][j]["choice"][k].lower()]
            for k in range(len(data[i][1][j]["choice"]), 4):
                d += ['']
            d += [data[i][1][j]["answer"].lower()] 
            D[0] += [d]
            break
        break

print(D[0][0][0])
print('\n')
print(D[0][0][1])
print('\n')
print(D[0][0][2])
print('\n')
print(len(D[0][0]))