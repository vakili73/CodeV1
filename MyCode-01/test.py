from Database import Info
from Loader import getDataset

for dataset in Info.keys():
    print(dataset)
    db = getDataset(dataset)

    demo = db.get_data(way=5, shot=5)
    print('')