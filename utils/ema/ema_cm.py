from utils.ema.default_ema import DefaultEMA 

class CMEMA(DefaultEMA):
    def __init__(
        self, 
        beta=0.9999, 
        update_every=1,
        update_after_step=1,
        is_distillation=False
        ):

        super().__init__(beta, update_every, update_after_step)
        self.is_distillation=is_distillation
        
    
    