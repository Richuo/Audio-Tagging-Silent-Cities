# Birds-Recognition
Download mp3 files, data processing and training functions for birds classification, using pretrained CNN models.

### Download audiofiles and process data
Download the metadata of the birds in your BirdsList
```
python create_birds_dataset.py
```

Download the audio files (mp3)
```
python download_birds_audio.py
```

Process the audio (convert mp3 to wav) and create labels csv
```
python mp3towav.py
```
(! The data_processing.py file is for data processing just before training !)

### Download pretrained models
Download your model from https://zenodo.org/record/3576403#.XqsOuyPVJPY and put it in the pretrained_models folder.

### Launch training (Transfer Learning)
Example: (for ResNet22)
```
python main.py --model_path="pretrained_models/ResNet22_mAP=0.430.pth" --labels_path="labels/all_data.csv" --trained_models_path="models" --graphs_path="graphs" --saving=1 --model_type="Transfer_ResNet22" --nb_species=13 --lr=1e-4 --batch_size=16 --epochs=40
```


model_path: Path to your pretrained model

saving: 0 = do not save; 1 = save training graph and trained model to .pth

model_type: Your Pytorch model in transflearn_models.py

nb_species: Your number of classes


### Credits
Author: Richard Huang

Dataset retrieved from https://www.xeno-canto.org

Download function inspired by [AgaMiko](https://github.com/AgaMiko/xeno-canto-download)

Code for Audioset Tagging CNN from [Qiu Qiang Kong](https://github.com/qiuqiangkong/audioset_tagging_cnn)
