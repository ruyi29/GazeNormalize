import pickle

with open('./data/pkl/test/subject01.pkl', 'rb') as fo:
    tinydict2 = pickle.load(fo, encoding='bytes')

print(tinydict2[0]['R'])
