from model.autoencoder import Encoder, Decoder

class AutoEncoder():
    def __init__(self, 
        config
    ):
        self.framework_config = config['framework']
        self.model_config = config['model']
        self.mode = self.framework_config['autoencoder']['type']
        self.encoder = Encoder(**self.model_config['autoencoder'])
        self.decoder = Decoder(**self.model_config['autoencoder'])
    
    def loss(self, mode):
        if mode == "VQ":
            pass
        if mode == "KL":
            pass
    
    def encoder_fit(self, x):
        return self.encoder(x)
        
    def decoder_fit(self, x):
        return self.decoder(x)
