
from modules.convlstm import ConvLSTM
import torch

def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set torch to use float16
    torch.set_default_dtype(torch.float16)

    input_dim = 3 # we have 3 channels for RGB
    kernel_size = 3 # kernel size
    num_classes = 13 # number of output classes
    num_layers = 3 # number of ConvLSTM layers
    hidden_dims = [16, 64, 256] # hidden dimensions for each ConvLSTM layer

    height, width = 512, 288  # input image size

    # Create the model
    model = ConvLSTM(input_dim, hidden_dims, kernel_size, num_classes, num_layers)
    model.to(device)
    # count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())

    print(f"Number of parameters: {num_params}")
    
    print(model)
    # Create a random input tensor
    batch_size = 1
    seq_len = 10

    input_tensor = torch.rand(batch_size, seq_len, input_dim, height, width).to(device)
    print(input_tensor.shape)

    model.reset_last_hidden_states()
    
    outputs = model(input_tensor)
    print(outputs.shape)
    
    # Benchmark using 100 random inputs, and using the process_frame method to emulate real-time processing
    import tqdm


    # # reset last hidden state
    model.reset_last_hidden_states()
    detach_steps = 50

    for i in tqdm.tqdm(range(100)):
        input_tensor = torch.rand(batch_size, seq_len, input_dim, height, width).to(device)
        outputs = model(input_tensor)

    

if __name__ == "__main__":
    main()