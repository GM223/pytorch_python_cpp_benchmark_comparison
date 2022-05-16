#!/bin/sh
cd code/cpp/build/src/

echo CPP ResNet18 save to cpp_resnet18_out.csv
./resnet-cifar-demo cpp_resnet18_out.csv 1

echo CPP ResNet34 save to cpp_resnet34_out.csv
./resnet-cifar-demo cpp_resnet34_out.csv 2

# navigate back python directory
cd ..
cd ..
cd ..
cd python
echo Python ResNet18 save to python_resnet18_out.csv
python main.py -f python_resnet18_out.csv -m ResNet18

echo Python ResNet32 save to python_resnet32_out.csv
python main.py -f python_resnet32_out.csv -m ResNet34

