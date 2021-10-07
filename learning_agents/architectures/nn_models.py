import torch
import torch.nn as nn
import torch.nn.functional as F


from learning_agents.architectures.utils import identity, init_layer_uniform



class MLP(nn.Module):
    """
    Baseline of Multilayer perceptron. The layer-norm is not implemented here
    Attributes:
        input_size (int): size of input
        output_size (int): size of output layer
        hidden_sizes (list): sizes of hidden layers
        hidden_activation (function): activation function of hidden layers
        output_activation (function): activation function of output layer
        hidden_layers (list): list containing linear layers
        use_output_layer (bool): whether or not to use the last layer
        n_category (int): category number (-1 if the action is continuous)
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        hidden_activation=F.relu,
        output_activation=identity,
        linear_layer=nn.Linear,
        use_output_layer=True,
        n_category=-1,
        init_fn=init_layer_uniform
    ):
        """
        Initialize.
        Args:
            input_size (int): size of input
            output_size (int): size of output layer
            hidden_sizes (list): number of hidden layers
            hidden_activation (function): activation function of hidden layers
            output_activation (function): activation function of output layer
            linear_layer (nn.Module): linear layer of mlp
            use_output_layer (bool): whether or not to use the last layer
            n_category (int): category number (-1 if the action is continuous)
            init_fn (Callable): weight initialization function bound for the last layer
        """
        super(MLP, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.linear_layer = linear_layer
        self.use_output_layer = use_output_layer
        self.n_category = n_category

        # set hidden layers
        self.hidden_layers = []
        in_size = self.input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = self.linear_layer(in_size, next_size)
            in_size = next_size
            self.__setattr__("hidden_fc{}".format(i), fc)
            self.hidden_layers.append(fc)

        # set output layers
        if self.use_output_layer:
            self.output_layer = self.linear_layer(in_size, output_size)
            self.output_layer = init_fn(self.output_layer)
        else:
            self.output_layer = identity
            self.output_activation = identity

    def forward(self, x):
        """
        Forward method implementation.
        x: torch.Tensor
        :return: torch.Tensor
        """
        for hidden_layer in self.hidden_layers:
            x = self.hidden_activation(hidden_layer(x))
        x = self.output_activation(self.output_layer(x))

        return x


class CNNLayer(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride=1,
        padding=0,
        pre_activation_fn=identity,
        activation_fn=torch.relu,
        post_activation_fn=identity,
    ):
        super(CNNLayer, self).__init__()
        self.cnn = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.pre_activation_fn = pre_activation_fn
        self.activation_fn = activation_fn
        self.post_activation_fn = post_activation_fn

    def forward(self, x):
        x = self.cnn(x)
        x = self.pre_activation_fn(x)
        x = self.activation_fn(x)
        x = self.post_activation_fn(x)

        return x


class CNN(nn.Module):
    """ Baseline of Convolution neural network. """
    def __init__(self, cnn_layers, fc_layers):
        """
        cnn_layers: List[CNNLayer]
        fc_layers: MLP
        """
        super(CNN, self).__init__()

        self.cnn_layers = cnn_layers
        self.fc_layers = fc_layers

        self.cnn = nn.Sequential()
        for i, cnn_layer in enumerate(self.cnn_layers):
            self.cnn.add_module("cnn_{}".format(i), cnn_layer)

    def get_cnn_features(self, x, is_flatten=True):
        """
        Get the output of CNN.
        """
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        x = self.cnn(x)
        # flatten x
        if is_flatten:
            x = x.view(x.size(0), -1)
        return x

    def forward(self, x, **fc_kwargs):
        """
        Forward method implementation.
        x: torch.Tensor
        :return: torch.Tensor
        """
        x = self.get_cnn_features(x)
        x = self.fc_layers(x, **fc_kwargs)
        return x



class Conv2d_MLP_Model(nn.Module):
    """ Default convolution neural network composed of conv2d layer followed by fully-connected MLP models """

    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu,
                 fc_output_activation=identity
                 ):
        super(Conv2d_MLP_Model, self).__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [input_channels] + channels[:-1]

        post_activation_fns = [identity for _ in range(len(strides))]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            post_activation_fns = [torch.nn.MaxPool2d(max_pool_stride) for max_pool_stride in strides]
            strides = ones
        activation_fns = [nonlinearity for _ in range(len(strides))]

        conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
                                kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
                       for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, channels, kernel_sizes, strides, paddings,
                                                                activation_fns, post_activation_fns)]
        fc_layers = MLP(
            input_size=fc_input_size,
            output_size=fc_output_size,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation,
            output_activation=fc_output_activation
        )

        self.conv_mlp = CNN(cnn_layers=conv_layers, fc_layers=fc_layers)

    def forward(self, x):
        return self.conv_mlp.forward(x)

class DeConvLayer(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride=1,
        padding=0,
        pre_activation_fn=identity,
        activation_fn=torch.relu,
        post_activation_fn=identity,
    ):
        super(DeConvLayer, self).__init__()
        self.de_conv = nn.ConvTranspose2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.pre_activation_fn = pre_activation_fn
        self.activation_fn = activation_fn
        self.post_activation_fn = post_activation_fn

    def forward(self, x):
        x = self.de_conv(x)
        x = self.pre_activation_fn(x)
        x = self.activation_fn(x)
        x = self.post_activation_fn(x)
        return x


# noinspection PyDefaultArgument
class DeConv2d_Model(nn.Module):
    def __init__(self,
                 input_channel=64,
                 channels=[64, 32, 4],
                 kernel_sizes=[3, 4, 8],
                 strides=[1, 2, 4],
                 paddings=[0, 0, 1],
                 activation_fn=[torch.relu, torch.relu, torch.sigmoid],
                 post_activation_fn=[identity, identity, identity]):
        super(DeConv2d_Model, self).__init__()

        self.deconv_layers = nn.Sequential()
        for i, _ in enumerate(channels):
            in_channel = input_channel if i == 0 else channels[i-1]
            deconv_layer = DeConvLayer(input_channels=in_channel, output_channels=channels[i],
                                       kernel_size=kernel_sizes[i], stride=strides[i],
                                       padding=paddings[i], activation_fn=activation_fn[i],
                                       post_activation_fn=post_activation_fn[i])
            self.deconv_layers.add_module("deconv_{}".format(i), deconv_layer)

    def forward(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        deconv_x = self.deconv_layers.forward(x)
        return deconv_x


class Conv2d_Deconv_MLP(Conv2d_MLP_Model):
    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 64, 64],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[1, 0, 0],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu,
                 fc_output_activation=identity,
                 deconv_input_channel=64,
                 deconv_channels=[64, 32, 4],
                 deconv_kernel_sizes=[3, 4, 8],
                 deconv_strides=[1, 2, 4],
                 deconv_paddings=[0, 0, 0],
                 deconv_activation_fn=[torch.relu, torch.relu, torch.sigmoid],
                 ):
        super(Conv2d_Deconv_MLP, self).__init__(input_channels=input_channels,
                                                fc_input_size=fc_input_size,
                                                fc_output_size=fc_output_size,
                                                channels=channels, kernel_sizes=kernel_sizes, strides=strides,
                                                paddings=paddings, nonlinearity=nonlinearity,
                                                use_maxpool=use_maxpool, fc_hidden_sizes=fc_hidden_sizes,
                                                fc_hidden_activation=fc_hidden_activation,
                                                fc_output_activation=fc_output_activation)

        self.decoder = DeConv2d_Model(channels=deconv_channels,
                                      input_channel=deconv_input_channel,
                                      kernel_sizes=deconv_kernel_sizes,
                                      strides=deconv_strides,
                                      paddings=deconv_paddings,
                                      activation_fn=deconv_activation_fn)

    def encode(self, x):
        return self.conv_mlp.get_cnn_features(x, is_flatten=False)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.conv_mlp.forward(x)


if __name__ == '__main__':
    from torchsummary import summary

    model = Conv2d_MLP_Model(3, 288, 2)
    summary(model, (3, 30, 30))
