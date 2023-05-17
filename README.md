# SonoNeRF

### Installation

* Clone this repository.
* Install dependecies
```
pip install -r requirements.txt
```
* Install ImageMagick
```
sudo apt-get update
sudo apt-get install imagemagick
```
* To preprocess data run:
```
python preprocess.py --input data/rawdata/volunteer01.mp4 --output data/preprocessed_data
```
* To train Nerf:
```
python train.py
```
