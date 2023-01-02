
import os
import yaml

def get_exp_config(config):
    return config['exp']

def get_current_exp_dir(config):
    exp_config = get_exp_config(config)
    current_exp_dir = os.path.join(exp_config['exp_dir'], exp_config['exp_name'])
    return current_exp_dir

def get_checkpoint_dir(config):
    exp_config= get_exp_config(config)
    current_exp_dir = get_current_exp_dir(config)
    current_checkpoint_dir = os.path.join(current_exp_dir, exp_config['checkpoint_dir'])
    return current_checkpoint_dir

def get_sampling_dir(config):
    exp_config= get_exp_config(config)
    current_exp_dir = get_current_exp_dir(config)
    sampling_dir = os.path.join(current_exp_dir, exp_config['sampling_dir'])
    return sampling_dir

def get_in_process_dir(config):
    exp_config= get_exp_config(config)
    current_exp_dir = get_current_exp_dir(config)
    in_process_dir = os.path.join(current_exp_dir, exp_config['in_process_dir'])
    return in_process_dir

def verifying_or_create_workspace(config):
    exp_config = get_exp_config(config)
    current_exp_dir = get_current_exp_dir(config)

    # Creating current exp dir
    if not os.path.exists(current_exp_dir):
        os.makedirs(current_exp_dir)
        print("Creating experiment dir")
    
    # Creating config file
    config_filepath = os.path.join(current_exp_dir, 'config.yml')
    with open(config_filepath, 'w') as f:
        yaml.dump(config, f)
    
    # Creating checkpoint dir
    checkpoint_dir = os.path.join(current_exp_dir, exp_config['checkpoint_dir'])
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print("Creating checkpoint dir")
    
    # Creating sampling dir
    sampling_dir = os.path.join(current_exp_dir, exp_config['sampling_dir'])
    if not os.path.exists(sampling_dir):
        os.makedirs(sampling_dir)
        print("Creating sampling dir")
    
    # Creating in_process dir
    in_process_dir = os.path.join(current_exp_dir, exp_config['in_process_dir'])
    if not os.path.exists(in_process_dir):
        os.makedirs(in_process_dir)
        print("Creating sampling dir")

def get_start_step_from_checkpoint(config):
    checkpoint_format = "checkpoint_"
    checkpoint_dir = get_checkpoint_dir(config)
    max_num = 0
    for content in os.listdir(checkpoint_dir):
        if checkpoint_format in content:
            _, num = content.split(checkpoint_format)
            num = int(num)
            if num > max_num:
                max_num = num
    return max_num