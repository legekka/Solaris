import torch
import torch.nn as nn
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda())

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size, num_classes, num_layers):
        """
        Initialize multi-layer ConvLSTM.

        :param input_dim: Number of channels in the input image
        :param hidden_dims: List of hidden dimensions for each ConvLSTM layer
        :param kernel_size: Kernel size for ConvLSTM cells
        :param num_classes: Number of output classes
        :param num_layers: Number of ConvLSTM layers
        """
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.num_layers = num_layers

        # if hidden_dims is not a list, make it a list
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers

        # Creating multiple ConvLSTM layers
        self.layers = nn.ModuleList()  # Use nn.ModuleList
        for i in range(self.num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            self.layers.append(ConvLSTMCell(input_dim=layer_input_dim,
                                            hidden_dim=hidden_dims[i],
                                            kernel_size=kernel_size,
                                            bias=True))
        
        # Output layer
        self.fc = nn.Linear(hidden_dims[-1], num_classes)

        # initialize hidden states
        self.reset_last_hidden_states()

    def forward(self, input_tensor):
        """
        Forward pass through multiple ConvLSTM layers and an output layer.

        :param input_tensor: Input tensor of shape (batch_size, seq_len, channels, height, width)
        """
        batch_size, seq_len, _, height, width = input_tensor.size()
        
        # Initialize a tensor to store the outputs for each timestep
        # Assuming output to be accumulated for the last layer
        outputs = torch.zeros(batch_size, seq_len, self.num_classes).cuda()

        # Forward pass through sequence for each layer
        for t in range(seq_len):
            x = input_tensor[:, t, :, :, :]
            for i, layer in enumerate(self.layers):
                # Update the state for each layer
                h, c = layer(x, self.last_hidden_states[i])
                # detach the hidden states from the graph
                h = h.detach()
                c = c.detach()

                self.last_hidden_states[i] = (h, c)
                x = h  # Output of current layer is input for the next

            # Use the output of the last layer for classification
            h_current = self.last_hidden_states[-1][0][:, :, :, :].mean(dim=[2, 3])  # Global average pooling

            # Fully connected layer
            output = self.fc(h_current)

            # Store the output for each timestep
            outputs[:, t, :] = output

        return outputs
    
    def reset_last_hidden_states(self, batch_size=1, image_size=(512, 288)):
        """
        Reset the last hidden states to zero.
        """
        self.last_hidden_states = [layer.init_hidden(batch_size, image_size) for layer in self.layers]
