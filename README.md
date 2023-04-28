# SonoNeRF

### Installation

* Clone this repository.
* Install dependecies
```
pip install -r requirements.txt
```
* To preprocess data run:
```
python3 preprocess.py --input data/rawdata/volunteer01.mp4 --output data/preprocesse
d_data
```

* To train Nerf:
```
python3 train.py --config configs/volunteer_01_short.txt
```