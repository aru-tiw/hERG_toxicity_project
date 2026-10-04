from tdc.single_pred import Tox
data = Tox(name = 'herg')
df = data.get_data()
df.to_csv('herg_dataset.csv', index=False)

