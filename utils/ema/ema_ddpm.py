from utils.ema.default_ema import DefaultEMA 


class DDPMEMA(DefaultEMA):
    def __init__(
        self,
        beta=0.9999, 
        update_every=1,
        update_after_step=1,
        ):
        super().__init__(beta, update_every, update_after_step)

if __name__=="__main__":
    import time
    count = 0
    sample_dict = {
        "beta": 0.5,
        "update_every": 1,
        "update_after_step": 0,
        "power": 2/3,
        "ema_rampup_ratio": 0.05,
        "ema_halflife_number": 500000
    }
    ema_obj = DDPMEMA(**sample_dict)
    while True:
        print(count)
        print(ema_obj.get_current_decay(count))
        print(ema_obj.get_power(count))
        time.sleep(0.1)
        count += 1
    