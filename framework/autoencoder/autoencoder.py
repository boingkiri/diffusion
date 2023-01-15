from model.autoencoder import Encoder, Decoder

def loss(mode):
    if mode == "VQ":
        pass
    if mode == "KL":
        pass


# Firstly, I implement autoencoder without any regularization such as VQ and KL.
# However, it should be implemented too someday..  
class AutoEncoder():
    def __init__(self, config):
        self.framework_config = config['framework']
        self.model_config = config['model']
        self.mode = self.framework_config['autoencoder']['type']
        self.encoder = Encoder(**self.model_config['autoencoder'])
        self.decoder = Decoder(**self.model_config['autoencoder'])
        self.loss_fn = 
    
    def encoder_fit(self, x):
        e_x = self.encoder(x)
        return e_x
        
    def decoder_fit(self, x):
        d_e_x = self.decoder(x)
        return d_e_x
