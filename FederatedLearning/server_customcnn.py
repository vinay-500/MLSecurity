import socket
import threading
import pickle
import torch
import torch.nn as nn
import torchvision

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input channels 3, output 32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Output size: 32 x 16 x 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output 64 x 16 x 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Output size: 64 x 8 x 8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output 128 x 8 x 8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Output size: 128 x 4 x 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Server:
    def __init__(self, host='localhost', port=5000, num_clients=2, num_rounds=5):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.client_sockets = []
        self.client_addresses = []
        self.model = self.initialize_model()
        self.lock = threading.Lock()
        self.client_data = {}

    def initialize_model(self):
        model = SimpleCNN(num_classes=10)
        return model

    def start(self):
        print("Starting server...")
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind((self.host, self.port))
        server_sock.listen()
        print(f"Server listening on {self.host}:{self.port}")

        # Accept clients until we have num_clients
        while len(self.client_sockets) < self.num_clients:
            client_sock, addr = server_sock.accept()
            print(f"Client connected from {addr}")
            self.client_sockets.append(client_sock)
            self.client_addresses.append(addr)

        print("All clients connected. Starting training rounds.")

        # Start training rounds
        for round_num in range(self.num_rounds):
            print(f"\nServer: Round {round_num+1}")

            # Collect model parameters from all clients
            self.collect_client_models()

            # Aggregate models
            self.aggregate_models()

            # Send updated global model to all clients
            self.send_global_model()

        print("Training completed.")
        
        torch.save(self.model.state_dict(), 'global_model.pth')

        # Close all client sockets
        for client_sock in self.client_sockets:
            client_sock.close()

        server_sock.close()

    def collect_client_models(self):
        self.client_data = {}
        for client_sock, addr in zip(self.client_sockets, self.client_addresses):
            print(f"Receiving model parameters from client {addr}")

            # Receive the length of the incoming data first
            data_length = client_sock.recv(16)
            data_length = int(data_length.decode('utf-8').strip())
            print(f"Server: Expecting {data_length} bytes from client {addr}")

            # Now receive the data in chunks
            data = b""
            received = 0
            while received < data_length:
                packet = client_sock.recv(4096)
                if not packet:
                    break
                data += packet
                received += len(packet)
            
            print(f"Server: Received {received} bytes from client {addr}")

            # Deserialize the model parameters
            if len(data) == data_length:
                client_model_params = pickle.loads(data)
                self.client_data[addr] = client_model_params
                print(f"Model parameters successfully received from client {addr}")
            else:
                print(f"Error: Expected {data_length} bytes but received {len(data)} bytes from client {addr}")

    def aggregate_models(self):
        # Aggregate the model parameters from clients using FedAvg
        with self.lock:
            print("Aggregating model parameters...")
            model_params_list = [params for params in self.client_data.values()]
            averaged_params = {}
            for key in model_params_list[0].keys():
                averaged_params[key] = sum([params[key] for params in model_params_list]) / len(model_params_list)
            self.model.load_state_dict(averaged_params)
            print("Model parameters aggregated.")

    def send_global_model(self):
        print("Sending updated global model to clients.")
        global_model_params = self.model.state_dict()
        serialized_params = pickle.dumps(global_model_params)
        data_length = len(serialized_params)
        data_length_str = str(data_length).encode('utf-8').ljust(16)

        for client_sock in self.client_sockets:
            client_sock.sendall(data_length_str)
            client_sock.sendall(serialized_params)
        print("Global model sent to all clients.")

if __name__ == "__main__":
    server = Server(host='localhost', port=5003, num_clients=1, num_rounds=5)
    server.start()
