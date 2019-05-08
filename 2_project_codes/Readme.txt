The codes included is the final version of prject. All the .ipynb flies can be loaded by jupyter notebook and their python version is saved under the "python codes" folder.
1. resize_training_image.ipynb
    This file is used to crop all original images into size 256x256 for training convenience.
2. May.ipynb or May.py is for generator pretraining using MSE loss. Codes should run in the same folder with training_data folder.
3. May_dis.ipynb or May_dis.py is for WGAN-GP training. Codes should run in the same folder with training_data folder.
4. testFile.ipynb or testFile.py is used to randomly pick an image and apply the mask on it, and finally generate a inpainting image using the trained network. Checkpoints folder must in the same folder for providing trained model.

Other important folder MUST included when run the codes:
1. checkpoints: saving all models, 219MB
2. training_data: images those cropped by resize_training_image.ipynb, 4.3GB


--For your check convenience, I wrote a bash file which will automatically download every data from my s3 repo and setup tensorflow environment on aws, so that you can run testFile.py to get a result yourself.
--Project.sh