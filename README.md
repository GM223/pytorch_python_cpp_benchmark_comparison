# Performance Comparison of Pytorch Python and LibTorch  C++ API with ResNet model with CIFAR dataset. 

## Introduction

Pytorch is primarily used through its python interface although most of the underlying high-performance code is written in C++. A C++ interface  for Pytorch is also available that exposes the code underlying codebase. There are many cases where a need arises for the use of C++ instead of the primary python API to meet project needs, such as in Low latency operation for robotics, stock trading etc.  

The goal of this project is to compare the performance of the two API and provide a quantitative comparison that can be used to make choices on when and where to use the different API’s.

The primary reason to use Pytorch with C++ are as following:
1. Low Latency Systems: For real time applications such as self-driving cars, game AI and other real-time projects the need lower latency, pure C++ is a much better fit here as compared to the latency limitations of the python interpreter.
2. Highly Multithreaded Environments: The Global Interpreter Lock (GIL) in python prevents it from running more than one thread at a time. Multiprocessing is available but it is not as scalable and has some shortcomings. C++ is not limited by such constraints so for application where the models require heavy penalization such as Deep Neuroevolution, this can benefit.
3. Existing C++ Codebases: Many existing projects have existing C++ code bases, and it is more convenient to keep everything in on language as supposed to binding C++ and python. Using C++ throughout the project is a more elegant solution.

## Directory Structure

/code has the cpp and python code in separate folders.
The docs folder contains the presentation, recorded video, results and calculations.

## Experiment Results and Conclusion

This experiment was conducted in a docker container running on an AMD Ryzen 9 5900HS (16 CPUs)~3.3GHz, 16 GB RAM and NVIDIA GeForce RTX 3070 Laptop GPU.

### Results:

| **Language and Model** | **Total excecution time (s)** | **Time per Epoch (s)** | **Inference Latency image/ms** | **Training Loss** | **Training Accuracy %** | **Test Loss** | **Test Accuracy %** |
| ---------------------- | ----------------------------- | ---------------------- | ------------------------------ | ----------------- | ----------------------- | ------------- | :-----------------: |
| **C++ ResNet18**       | 361.814                       | 7.235071992            | 3.31                           | 0.0948172         | 96.684                  | 1.34854       |        70.44        |
| **C++ ResNet34**       | 553.106                       | 11.06131683            | 4.23                           | 0.105326          | 96.406                  | 1.25164       |        72.06        |
| **Python ResNet18**    | 1289.118005                   | 25.77832627            | 2.30788256                     | 0.107002093       | 96.238                  | 0.449515      |        88.41        |
| **Python ResNet34**    | 2327.167102                   | 45.23919889            | 3.968456321                    | 0.119861529       | 95.83                   | 0.4285611     |        88.1         |

### Observations

•The LibTorch C++ program has more than **3-4x speedup in training and test time** as compared to the Pytorch implementation. This is possibly because it is compiled beforehand as well as there is no Global Interpreter Lock (GIL) in C++. 

•The latency counter intuitively is better with the python code, I am not sure why this is, possibly because latency is measured with a single image, and it only needs 1 thread.

•The Training accuracy for all the models in very similar at around 96% but the test accuracy for the C++ ones is around 71% as opposed to 88% for the Pytorch version. Not sure why this is going on, possible due to non identical data augmentation even though I checked the code for anything like that and could not find it. 

### Conclusion

Taken together, it is not recommended to use LibTorch C++ API unless there are some special use cases.

## Set up Docker Container

```
$ docker pull nvcr.io/nvidia/pytorch:21.03-py3
$ docker run -it --rm --gpus all --ipc=host -v $(pwd):/mnt nvcr.io/nvidia/pytorch:21.03-py3

For window, change your path to the cloned directory in the comand below

docker run -it --rm --gpus all --ipc=host -v C:/Users/you/yourfolder/Education/NYU/Spring_2022/High_Performance_Machine_Learning/project/pytorch_python_cpp_benchmark_comparison:/workspace/project nvcr.io/nvidia/pytorch:21.03-py3
```

## Using C++

### Download Dataset

```
$ cd /code/cpp
$ mkdir -p dataset
$ cd dataset/
$ bash download-cifar10-binary.sh
```

### Build Application

```
$ cmake -B build -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
$ cmake --build build --config Release
```

### Run Application

First argument is filename output is stored to, the second is ResNet size. 1 is for ResNet18 and 2 is for ResNet34

```
$ cd build/src/
$ ./resnet-cifar-demo filename.csv 1

echo CPP ResNet18 save to cpp_resnet18_out.csv
./resnet-cifar-demo cpp_resnet18_out.csv 1

echo CPP ResNet34 save to cpp_resnet34_out.csv
./resnet-cifar-demo cpp_resnet34_out.csv 2
```

The output files are in code/cpp/build/src folder.

## Using Python

First argument -f is filename, the second is -m model ResNet size. 1 is for ResNet18 and 2 is for ResNet34

```
$ cd code/python/
$ python main.py -f filename.csv -m ResNet18

#Example
echo Python ResNet18 save to python_resnet18_out.csv
python main.py -f python_resnet18_out.csv -m ResNet18

echo Python ResNet32 save to python_resnet32_out.csv
python main.py -f python_resnet32_out.csv -m ResNet34
```

## Using Bash script

The bash scrip is used to execute and log all 4 of the experiments serially with a single command. 

```
# Cd to the base directory pytorch_python_cpp_benchmark_comparison
$ bash benchmark.sh
```

## References and links:

•Lei Mao blog and git repository on LibTorch resnet
 https://leimao.github.io/blog/LibTorch-ResNet-CIFAR/
 https://github.com/leimao/LibTorch-ResNet-CIFAR

•Pytorch CIFAR10
 https://github.com/kuangliu/pytorch-cifar

•Pytorch NN latency 
 https://deci.ai/blog/measure-inference-time-deep-neural-networks/
