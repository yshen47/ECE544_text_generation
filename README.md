# FashionSentenseGeneration# ECE544_finalproject

### Folder contents
dataset/ contains dataset and preparation code
cumulative_attention/ contains all the code for data preparation, model class, training and testing

### Environment
We used python 3.5, pytorch 0.4.1, tqdm 4.26.0, so make sure these packages are installed in your environment.

### Run code
To train the model, simply run
```
python train.py
```
model will be saved to cumulative_attention/models/ file after every epoch 

To run test and redirct ouput to a result file in cumulative_attention/results/ folder, run
```
python test.py > cumulative_attention/results/[result file name]
```

The model type can be changed inside both train.py and test.py

### Other information
We saved pre-trained lstm and gru models in the cumulative_attention/models/ folder, and saved pre-ran results in cumulative_attention/results/ folder
