## model.py
***We use VGG16 as features extracter , which accepts input shape (2048,64,3). Before data flows through LSTM layers , it goes by a attention layer. We set up 2 modes. The first one is for training phase , where inputs are 4 arrays and output is loss function array . The second one is for prediction , where input is image array only and output is our prediction. You guys can type : python model.py to see summary of our model***

## load_data.py
***We set up a class which provides some methods to help us build data for feeeding model.Also including some data augmentations.Actually we feed model in batch thanks to next_bacth() method.It is just a supporting module. We will not working on it***

## train.py
***Here you guys can train our model.You guys can follow some instructions to adjust hyperparameters as desire.For example if you guys type : python train.py --train --label 10 32 0.0001 dinh_nana . That means you guys wanna train model with 10 epochs ,batch_size 32 , learning rate 0.0001 and name of model saved is dinh_nana.h5***

## predict.py
***Here you guys can see how good our model predict . You guys can type : python predict.py ./model/name.h5 --test_folder --test_label_path , which means you guys first load weights from model having the name you guys gave before , test_folder and test_label_path is set up in default. Or you guys can type --weight_path instead in order to use our pre - trained model*** 

## evaluate.py
***Here you guys can find a function to compute some errors like WER , CER , SER.That means it is just also supporting module. We will not also working on it***

## data/raw
***Here you guys need to upload data to this folder. Also note that you guys attach file labels.json,which contains ground truth***

## model
***Here the place our model will be saved. We also prepare a pre - trained model for you guys if you guys are so lazy to train something new***

## private_test
***Here you guys upload data used for predicting.Also remember that file labels.json is included***