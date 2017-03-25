from ml_ops import *

def baseline_model(input):
    """
    Run through the feed forward
    3D Convolution Neural Network.

    Layers:
        3D ConvNet Layers w/ PRelu followed by 3D Maxpool

        Fully Connected Layers w/ PRelu

        FC Output Layer that provides the single logits unit
    """
    
    h_conv1 = convolve(input, 1, 4, 'conv1')
    h_conv2 = convolve(h_conv1, 4, 4, 'conv2')
    h_conv3 = convolve(h_conv2, 4, 4, 'conv3')
    h_conv4 = convolve(h_conv3, 4, 4, 'conv4')
    h_conv5 = convolve(h_conv4, 4, 4, 'conv5')
    h_conv6 = convolve(h_conv5, 4, 4, 'conv6')
    
    print("Transfer layer shape: {}".format(h_conv6.get_shape()))
    
    h_fc1 = fc_layer(h_conv6, 32, 'fc1')
    output_logits = fc_layer(h_fc1, 1, 'fc2', activate=False)
        
    return output_logits


def larger_first_model(input):
    """
    Run through the feed forward
    3D Convolution Neural Network.

    Layers:
        3D ConvNet Layers w/ PRelu followed by 3D Maxpool

        Fully Connected Layers w/ PRelu

        FC Output Layer that provides the single logits unit
    """
    
    h_conv1 = convolve(input, 1, 8, 'conv1')
    h_conv2 = convolve(h_conv1, 8, 16, 'conv2')
    h_conv3 = convolve(h_conv2, 16, 16, 'conv3')
    h_conv4 = convolve(h_conv3, 16, 32, 'conv4')
    h_conv5 = convolve(h_conv4, 32, 64, 'conv5')
    h_conv6 = convolve(h_conv5, 64, 128, 'conv6')
    
    print("Transfer layer shape: {}".format(h_conv6.get_shape()))
    
    h_fc1 = fc_layer(h_conv6, 32, 'fc1')
    output_logits = fc_layer(h_fc1, 1, 'fc2', activate=False)
        
    return output_logits

