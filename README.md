# PyTorch MNIST example


## Prerequisites

- PyTorch 1.5.0
- TorchVision 0.6.0


## Training

    $ python train.py \
        --train_batch_size=64 \
        --test_batch_size=50 \
        --epochs=5 \
        --lr=1e-3 \
        --log_interval=100 \
        --output_model_name="models/mnist_cnn.pt"


## Evaluation

    $ python evaluate.py \
        --model_path="models/mnist_cnn.pt" \
        --batch_size=50


## Prediction

    $ python predict.py \
        --model_path="models/mnist_cnn.pt" \
        --image=<path/to/image>


## Running gRPC Server

1. Generate Python code from `digit_classification.proto`, which provide gRPC server and client interfaces.

        $ python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./digit_classification.proto

2. Run server example.

    To use this server, you have to also implement its client satisfying `digit_classification.proto`.

        $ python server.py \
            --model_path="./models/mnist_cnn.pt" \
            --max_workers=1
