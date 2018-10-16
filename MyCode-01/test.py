from Database import Info
from Loader import getDataset

for dataset in Info.keys():
    print(dataset)
    getDataset(dataset)