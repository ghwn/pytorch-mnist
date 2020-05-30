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
