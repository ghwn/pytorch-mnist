from concurrent import futures
import argparse
import random

import grpc

import digit_classification_pb2
import digit_classification_pb2_grpc
import predict as prediction_module


class DigitClassificationServicer(digit_classification_pb2_grpc.DigitClassificationServicer):
    """Provides methods that implement functionality of digit classification server."""

    def __init__(self, model_path):
        self.model_path = model_path

    def predict(self, request, context):
        image = request.data
        prediction = prediction_module.predict(self.model_path, image)
        return digit_classification_pb2.Prediction(data=prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="path to MNIST model")
    parser.add_argument('--max_workers', type=int, default=1, help="the number of max workers")
    args = parser.parse_args()

    server = grpc.server(thread_pool=futures.ThreadPoolExecutor(max_workers=args.max_workers))
    digit_classification_pb2_grpc.add_DigitClassificationServicer_to_server(
        servicer=DigitClassificationServicer(args.model_path), server=server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
