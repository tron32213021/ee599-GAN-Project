#!/bin/bash
cd /home/ubuntu
mkdir ee599_Project
cd ee599_Project
wget https://s3.us-east-2.amazonaws.com/ee599/Final_Project_package.zip
wget https://s3.us-east-2.amazonaws.com/ee599/Project_training_data.zip
unzip Final_Project_package.zip
unzip Project_training_data.zip
source activate tensorflow_p36