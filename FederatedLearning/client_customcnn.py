import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

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

class Client:
    def __init__(self, host='localhost', port=5000, num_rounds=5, client_id=0, num_clients=2):
        self.host = host
        self.port = port
        self.num_rounds = num_rounds
        self.client_id = client_id
        self.num_clients = num_clients
        self.model = self.initialize_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.trainloader = self.load_data()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def initialize_model(self):
        model = SimpleCNN(num_classes=10)
        return model

    def load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

        # Reduce dataset size by using only 100 samples per client for quicker debugging
        subset_size = 100
        client_subset = torch.utils.data.Subset(trainset, range(subset_size))
        
        trainloader = torch.utils.data.DataLoader(client_subset, batch_size=32, shuffle=True, num_workers=2)
        
        return trainloader  

    def train(self, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        print("Client training complete for this round.")

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print("Connected to server.")

    def send_model(self):
        model_params = self.model.state_dict()
        serialized_params = pickle.dumps(model_params)
        
        # Send length of data first
        data_length = len(serialized_params)
        self.sock.sendall(str(data_length).encode('utf-8').ljust(16))
        
        print(f"Client {self.client_id}: Sending model of size {data_length} bytes to server.")

        # Send data in chunks
        sent = 0
        while sent < data_length:
            chunk_size = 4096
            self.sock.sendall(serialized_params[sent:sent+chunk_size])
            sent += chunk_size

        print(f"Client {self.client_id}: Model sent to server.")    

    def receive_model(self):
        # Receive length of data first
        data_length = self.sock.recv(16)
        data_length = int(data_length.decode('utf-8').strip())
        # Now receive the data
        data = b""
        while len(data) < data_length:
            packet = self.sock.recv(4096)
            if not packet:
                break
            data += packet
        # Deserialize model parameters
        global_model_params = pickle.loads(data)
        self.model.load_state_dict(global_model_params)
        print("Received global model from server.")

    def run(self):
        self.connect()
        for round_num in range(self.num_rounds):
            print(f"\nClient {self.client_id}: Round {round_num+1}")
            # Train local model
            self.train(epochs=1)

            # Send model parameters to server
            self.send_model()
            print("Sent local model to server.")

            # Receive updated global model from server
            self.receive_model()

        self.sock.close()
        print("Training completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=int, default=0)
    parser.add_argument('--num_clients', type=int, default=2)
    args = parser.parse_args()
    client = Client(host='localhost', port=5003, num_rounds=5, client_id=args.client_id, num_clients=args.num_clients)
    client.run()
