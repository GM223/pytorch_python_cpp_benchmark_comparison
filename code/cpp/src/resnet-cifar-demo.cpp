#include <ATen/cuda/CUDAContext.h>
#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>
#include <c10/cuda/CUDAStream.h>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <torch/script.h>
#include <torch/torch.h>
#include <tuple>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include "cifar10.h"
#include "resnet.h"

template <typename T>
float measure_inference_latency(std::shared_ptr<T> model, torch::Device& device,
                                const std::vector<int64_t>& input_size,
                                const int num_samples = 100,
                                const int num_warmups = 10)
{
    torch::NoGradGuard no_grad;
    model->eval();
    model->to(device);

    torch::Tensor inputs = torch::ones(input_size).to(device);

    int64_t batch_size = input_size.at(0);

    for (int i = 0; i < num_warmups; i++)
    {
        torch::Tensor output = model->forward(inputs);
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    for (int i = 0; i < num_samples; i++)
    {
        torch::Tensor output = model->forward(inputs);
        stream = at::cuda::getCurrentCUDAStream();
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();

    float avg_latency =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
            .count() /
        static_cast<float>(num_samples);

    return avg_latency;
}

template <typename T1, typename T2, typename T3>
std::tuple<float, float, float, float> evaluate_model(std::shared_ptr<T1> model,
                                        T2& test_data_loader, T3& criterion,
                                        torch::Device& device)
{
    // https://pytorch.org/cppdocs/api/typedef_namespacetorch_1abf2c764801b507b6a105664a2406a410.html#typedef-torch-nogradguard
    torch::NoGradGuard no_grad;
    model->eval();
    model->to(device);

    int64_t num_running_corrects = 0;
    int64_t num_samples = 0;
    float running_loss = 0;

    float total_batchLoad_time{0.0};
    float total_inference_time{0.0};
    
    // Iterate the data loader to yield batches from the dataset.
    for (torch::data::Example<>& batch : *test_data_loader)
    {
        auto start_batchLoad = std::chrono::high_resolution_clock::now();
        torch::Tensor inputs = batch.data.to(device);
        torch::Tensor labels = batch.target.to(device);
        auto stop_batchLoad = std::chrono::high_resolution_clock::now();
        total_batchLoad_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop_batchLoad - start_batchLoad).count();
        
        auto inference_time_start = std::chrono::high_resolution_clock::now();
        torch::Tensor outputs = model->forward(inputs);
        auto inference_time_stop = std::chrono::high_resolution_clock::now(); 
        total_inference_time += std::chrono::duration_cast<std::chrono::milliseconds>(inference_time_stop - inference_time_start).count();

        torch::Tensor preds = std::get<1>(torch::max(outputs, 1));
        num_running_corrects += torch::sum(preds == labels).item<int64_t>();
        // A related GCC bug:
        // https://discuss.pytorch.org/t/convert-c10-scalar-to-int/120513/2
        torch::Tensor loss_tensor = criterion(outputs, labels);
        float loss = loss_tensor.item<float>();
        num_samples += inputs.size(0);
        running_loss += loss * inputs.size(0);
    }

    float eval_accuracy =
        static_cast<float>(num_running_corrects*100) / num_samples;
    float eval_loss = running_loss / num_samples;

    return {eval_accuracy, eval_loss, total_inference_time, total_batchLoad_time};
}

template <typename T1, typename T2, typename T3>
std::shared_ptr<T1>
train_model(std::shared_ptr<T1> model, T2& train_data_loader,
            T3& test_data_loader, torch::Device& device,
            float learning_rate = 1e-1, int64_t num_epochs = 200, std::ofstream& outfile = 0)
{
    model->train();
    model->to(device);

    torch::nn::CrossEntropyLoss criterion{};

    // SGD optimizer
    // https://pytorch.org/cppdocs/api/structtorch_1_1optim_1_1_s_g_d_options.html#_CPPv4N5torch5optim10SGDOptionsE
    torch::optim::SGD optimizer{model->parameters(),
                                torch::optim::SGDOptions(/*lr=*/learning_rate)
                                    .momentum(0.9)
                                    .weight_decay(1e-4)};
    // Requires LibTorch >= 1.90
    // torch::optim::LRScheduler scheduler{optimizer, 50, 0.1};

    model->eval();

    std::cout << std::fixed;

    std::tuple<float, float, float, float> eval_result = evaluate_model(model, test_data_loader, criterion, device);
    std::cout << std::setprecision(6) << "Epoch: " << std::setfill('0')
              << std::setw(3) << 0 << " Eval Loss: " << std::get<1>(eval_result)
              << " Eval Acc: " << std::get<0>(eval_result) << std::endl;
              
    auto everthing_time_start = std::chrono::high_resolution_clock::now();
    float previous_wall_time{0.0};

    for (size_t epoch = 1; epoch <= num_epochs; epoch++)
    {
        model->train();

        int64_t num_running_corrects = 0;
        int64_t num_samples = 0;
        float running_loss = 0;
        float total_batchLoad_time{0.0};
        float total_inference_time{0.0};
        float total_training_time{0.0};
        
        // Iterate the data loader to yield batches from the dataset.
        for (torch::data::Example<>& batch : *train_data_loader)
        {
            auto start_batchLoad = std::chrono::high_resolution_clock::now();
            torch::Tensor inputs = batch.data.to(device);
            torch::Tensor labels = batch.target.to(device);
            auto stop_batchLoad = std::chrono::high_resolution_clock::now();
            total_batchLoad_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop_batchLoad - start_batchLoad).count();
            // Reset gradients.
            auto start_optimizer = std::chrono::high_resolution_clock::now();
            optimizer.zero_grad();
            // Execute the model on the input data.
            auto inference_time_start = std::chrono::high_resolution_clock::now();
            torch::Tensor outputs = model->forward(inputs);
            auto inference_time_stop = std::chrono::high_resolution_clock::now();
            total_inference_time += std::chrono::duration_cast<std::chrono::milliseconds>(inference_time_stop - inference_time_start).count();
            // Compute a loss value to judge the prediction of our model.
            torch::Tensor loss = criterion(outputs, labels);
            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();
            // Update the parameters based on the calculated gradients.
            optimizer.step();
            auto stop_optimizer = std::chrono::high_resolution_clock::now();
            total_training_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop_optimizer-start_optimizer).count(); 

            torch::Tensor preds = std::get<1>(torch::max(outputs, 1));
            num_running_corrects += torch::sum(preds == labels).item<int64_t>();
            num_samples += inputs.size(0);
            running_loss += loss.item<float>() * inputs.size(0);
        }

        float train_accuracy =
            static_cast<float>(num_running_corrects*100) / num_samples;
        float train_loss = running_loss / num_samples;

        model->eval();

        std::tuple<float, float, float, float> eval_result =
            evaluate_model(model, test_data_loader, criterion, device);

        float wall_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-everthing_time_start).count();

        std::cout << std::setprecision(6) << "Epoch: " << std::setfill('0')
                  << std::setw(3) << epoch << " Time s: "<< wall_time/1000 <<" Epoch Time s: "<<(wall_time-previous_wall_time)/1000<<" Train Loss: " << train_loss
                  << " Train Acc: " << train_accuracy
                  << " Eval Loss: " << std::get<1>(eval_result)
                  << " Eval Acc: " << std::get<0>(eval_result) << std::endl;

        
        outfile << std::setprecision(6) << epoch <<","<< wall_time/1000 <<","<<(wall_time-previous_wall_time)/1000<<","
                << train_loss <<","<< train_accuracy <<","
                << std::get<1>(eval_result) <<","<< std::get<0>(eval_result) <<","
                << std::endl;
        // scheduler.step();
        previous_wall_time = wall_time;
    }

    return model;
}

int main(int argc,char* argv[])
{
    if (argc != 3) {
        std::cerr << "Syntax: " << argv[0] << " <text file name>\n";
        return 1;
    }
    std::ofstream outfile{argv[1]};
    if (! outfile.is_open ()) { return EXIT_FAILURE ; }

    outfile<<"Epoch, Wall Time, Epoch time, Train_loss, Train_top_1, Test_loss, Test_top_1\r\n";

    const int64_t random_seed{0};
    const float learning_rate{1e-1};
    const int64_t num_epochs{50};
    torch::manual_seed(random_seed);
    // torch::cuda::manual_seed(random_seed);
    const int64_t batch_size{128};
    const int64_t num_workers{4};
    const std::string dataset_root{"../../dataset/cifar-10-batches-bin"};
    std::filesystem::path model_dir{"../../saved_models"};
    std::filesystem::path model_file_name{"resnet.pt"};
    std::filesystem::path model_file_path = model_dir / model_file_name;

    std::filesystem::create_directories(model_dir);

    CIFAR10 train_set{dataset_root, CIFAR10::Mode::kTrain};
    CIFAR10 test_set{dataset_root, CIFAR10::Mode::kTest};

    torch::Device device("cpu");
    if (torch::cuda::is_available())
    {
        device = torch::Device("cuda:0");
    }

    // This might be different from the PyTorch API.
    // We did transform for the dataset directly instead of doing transform in
    // dataloader. Currently there is no augmentation options such as random
    // crop.
    auto train_set_transformed =
        train_set
            .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465},
                                                      {0.2023, 0.1994, 0.2010}))
            .map(torch::data::transforms::Stack<>());

    auto test_set_transformed =
        test_set
            .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465},
                                                      {0.2023, 0.1994, 0.2010}))
            .map(torch::data::transforms::Stack<>());

    auto train_data_loader = torch::data::make_data_loader(
        std::move(train_set_transformed), torch::data::DataLoaderOptions()
                                              .batch_size(batch_size)
                                              .workers(num_workers)
                                              .enforce_ordering(true));

    auto test_data_loader = torch::data::make_data_loader(
        std::move(test_set_transformed), torch::data::DataLoaderOptions()
                                             .batch_size(batch_size)
                                             .workers(num_workers)
                                             .enforce_ordering(true));
                                    
    std::shared_ptr<ResNet<BasicBlock>> model{};
    
    int val = std::stoi(argv[2]);
    std::cout<<val<<std::endl;
    if (val== 1){
        std::cout<<"Model:ResNet18"<<std::endl;
        model = resnet18(/*num_classes = */ 10);
    }
    else if (val == 2)
    {
        std::cout<<"Model:ResNet32"<<std::endl;
        model = resnet34(/*num_classes = */ 10);
    }

    

    std::cout << "Training Model..." << std::endl;
    model = train_model(model, train_data_loader, test_data_loader, device,
                        learning_rate, num_epochs, outfile);
    std::cout << "Training Finished." << std::endl;

    torch::save(model, model_file_path);
    torch::load(model, model_file_path);

    const std::vector<int64_t> input_size{1, 3, 32, 32};
    std::cout << "Measuring Latency..." << std::endl;
    float latency =
        measure_inference_latency(model, device, input_size, 100, 10);
    outfile<<"Latency,"<<latency<<" in ms"<<std::endl;
    std::cout << "Inference Latency (BS = " << input_size.at(0)
              << "): " << std::setprecision(4) << latency << " [ms / image]"
              << std::endl;
}
