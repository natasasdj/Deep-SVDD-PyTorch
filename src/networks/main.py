from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder 
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .fmnist_LeNet import FMNIST_LeNet, FMNIST_LeNet_Autoencoder 

def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'fmnist_LeNet' )
    assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    if net_name == 'fmnist_LeNet':
        net = FMNIST_LeNet()


    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'fmnist_LeNet')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'fmnist_LeNet':
        ae_net = FMNIST_LeNet_Autoencoder()

    return ae_net
