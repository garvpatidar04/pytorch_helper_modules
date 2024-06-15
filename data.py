import os
import zipfile
import requests
from pathlib import Path

data_path = Path('data')

main_path = data_path / 'pizza_sushi_steak'

url_ = 'https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip'

if main_path.is_dir():
    print('skipping creating the folder')
else:
    main_path.mkdir(parents=True, exist_ok=True)
    print('Creating the folder...')

with open(data_path/'pizza_steak_sushi.zip', 'wb') as f:
    request = requests.get(url=url_, timeout=10) 
    f.write(request.content)
    print('Downloading the data...')

with zipfile.ZipFile(data_path/'pizza_steak_sushi.zip', 'r') as zip_ref:
    print('Unzipping the data...')
    zip_ref.extractall(main_path)

os.remove(data_path/'pizza_steak_sushi.zip')
print('Removed the zip file')