import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from argparse import ArgumentParser, ArgumentTypeError

def generate_linear(n=100):
    inputs, labels = [], []
    pts = np.random.uniform(0, 1, (n, 2))

    for pt in pts:
        inputs.append([pt[0], pt[1]])
        labels.append(int(pt[0] <= pt[1]))
        
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy(n=11):
    inputs, labels = [], []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue

        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)

class Layer:
    def __init__(self, input_link: int, output_link: int, activation: str, learning_rate: float, derivation=None) -> None:
        self.learning_rate = learning_rate
        self.derivation = self.__set_derivation(activation, derivation)
        self.activation = self.__set_activation(activation)
        self.weight = np.random.normal(0, 1, (input_link + 1, output_link))

    def __set_activation(self, activation: str) -> Callable:
        if isinstance(activation, Callable):
            return activation
        elif isinstance(activation, str):
            try:
                return eval(f'self.{activation}')
            except:
                raise ValueError(f'Does not support activate function {activation}')
        else:
            raise TypeError(f'Does not support type {type(activation)} for activate function')
        
    def __set_derivation(self, activation: str, derivation: Callable) -> Callable:
        if isinstance(activation, str):
            try:
                return eval(f'self.derivative_{activation}')
            except:
                raise ValueError(f'Does not support derivative function {activation}')
        else:
            if derivation == None:
                return self.derivative_empty
            else:
                return derivation

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.forward_gradient = np.append(inputs, np.ones((inputs.shape[0], 1)), axis=1)
        self.outputs = self.activation(np.matmul(self.forward_gradient, self.weight))

        return self.outputs
    
    def backward(self, derivative_loss: np.ndarray) -> np.ndarray:
        if np.array_equal(derivative_loss, self.activation(derivative_loss)):
            self.backward_gradient = derivative_loss
        else:
            self.backward_gradient = np.multiply(self.derivation(self.outputs), derivative_loss)
        return np.matmul(self.backward_gradient, self.weight[:-1].T)
    
    def update(self) -> None:
        gradient = np.matmul(self.forward_gradient.T, self.backward_gradient)
        delta_weight = -self.learning_rate * gradient
        self.weight += delta_weight

    @staticmethod
    def empty(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def derivative_empty(y: np.ndarray) -> np.ndarray:
        return y

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def derivative_sigmoid(y: np.ndarray) -> np.ndarray:
        return np.multiply(y, 1.0 - y)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def derivative_tanh(y: np.ndarray) -> np.ndarray:
        return 1.0 - y ** 2
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def derivative_relu(y: np.ndarray) -> np.ndarray:
        return np.heaviside(y, 0)
    
class NeuralNetwork:
    def __init__(
            self,
            epoch: int,
            learning_rate: float,
            num_of_layers: int,
            input_units: int,
            hidden_units: int,
            activation: str,
            ) -> None:
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.activation = activation
        self.learning_epoch, self.learning_loss = [], []

        # input layer
        self.layers = [Layer(input_units, hidden_units, activation, learning_rate)]
        # hidden layer
        for _ in range(num_of_layers):
            self.layers.append(Layer(hidden_units, hidden_units, activation, learning_rate))
        # output layer
        self.layers.append(Layer(hidden_units, 1, 'sigmoid', learning_rate))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs
    
    def backward(self, derivative_loss: np.ndarray) -> None:
        for layer in self.layers[::-1]:
            derivative_loss = layer.backward(derivative_loss)

    def update(self) -> None:
        for layer in self.layers:
            layer.update()

    def mse_loss(self, prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        return np.mean((prediction - ground_truth) ** 2)
    
    def mes_derivative_loss(self, prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        return 2 * (prediction - ground_truth) / len(ground_truth)

    def train(self, inputs: np.ndarray, labels: np.ndarray) -> None:
        for cur_epoch in range(self.epoch):
            predict = self.forward(inputs)
            loss = self.mse_loss(predict, labels)
            self.backward(self.mes_derivative_loss(predict, labels))
            self.update()

            if cur_epoch % 100 == 0:
                print(f'Epoch {cur_epoch} loss: {loss}')
                self.learning_epoch.append(cur_epoch)
                self.learning_loss.append(loss)

            if loss < 0.001:
                return
            
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        prediction = self.forward(inputs)
        print(prediction)
        return np.round(prediction)
    
    def show_result(self, data_set: str, inputs: np.ndarray, labels: np.ndarray) -> None:
        pred_labels = self.predict(inputs)

        print(f'Activation : {self.activation}')
        print(f'Hidden units : {self.hidden_units}')
        print(f'Accuracy : {float(np.sum(pred_labels == labels)) / len(labels)}')

        plt.figure()
        plt.subplot(2, 2, 1)
        plt.title('Ground truth', fontsize=18)
        for idx, point in enumerate(inputs):
            plt.plot(point[0], point[1], 'ro' if labels[idx][0] == 0 else 'bo')

        plt.subplot(2, 2, 2)
        plt.title('Predict result', fontsize=18)
        for idx, point in enumerate(inputs):
            plt.plot(point[0], point[1], 'ro' if pred_labels[idx][0] == 0 else 'bo')
        plt.savefig(f'{data_set}_result.png')

        plt.figure()
        plt.title('Learning curve', fontsize=18)
        plt.plot(self.learning_epoch, self.learning_loss)

        plt.savefig(f'{data_set}_learning.png')

def parse_argument():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_set', default='linear', type=str, help='linear data points or XOR data points')
    parser.add_argument('-n', '--number_of_data', default=100, type=int, help='Number of data points')
    parser.add_argument('-e', '--epoch', default=10000, type=int, help='Number of epch')
    parser.add_argument('-l', '--learning_rate', default=0.5, type=float, help='Learning rate')
    parser.add_argument('-u', '--hidden_unit_num', default=4, type=int, help='Number of hidden units')
    parser.add_argument('-a', '--activation', default='sigmoid', type=str, help='Type for activation function')
    
    return parser.parse_args()

def main():
    args = parse_argument()
    data_set = args.data_set
    number_of_data = args.number_of_data
    epoch = args.epoch
    learning_rate = args.learning_rate
    hidden_unit_num = args.hidden_unit_num
    activation = args.activation

    if data_set == 'linear':
        inputs, labels = generate_linear(number_of_data)
    elif data_set == 'xor':
        inputs, labels = generate_XOR_easy(number_of_data)

    neural_network = NeuralNetwork(
        epoch=epoch,
        learning_rate=learning_rate,
        num_of_layers=2,
        input_units=2,
        hidden_units=hidden_unit_num,
        activation=activation
    )

    neural_network.train(inputs=inputs, labels=labels)
    neural_network.show_result(data_set, inputs=inputs, labels=labels)

if __name__ == '__main__':
    main()