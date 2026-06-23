from ml_collections import ConfigDict
import yaml


def load_configs(file, additional_cfg = None):
    with open(file) as f:
        cfg_file = yaml.safe_load(f)

    cfgs = [ConfigDict(cfg_file)]
    while "parent" in cfgs[-1] and cfgs[-1].parent:
        with open(cfgs[-1].parent) as f:
            cfg_file = yaml.safe_load(f)

        cfgs.append(ConfigDict(cfg_file))

    cfg = cfgs.pop()
    while cfgs:
        cfg.update(cfgs.pop())
    
    if additional_cfg:
        i = 0
        while i < len(additional_cfg):
            if "=" in additional_cfg[i]:
                key, value = additional_cfg[i].split("=")
                key = key.lstrip('-')  
                i += 1
            else:
                key = additional_cfg[i].lstrip('-')  
                value = additional_cfg[i+1]
                i += 2
                
            if "." in key:
                keys = key.split(".")
            else:
                keys = [key]

            try:
                d = {keys.pop():eval(value)}
            except (NameError, SyntaxError):
                raise ValueError(
                    f'''
                        Invalid value for {key}: {value}. 
                        Please ensure it is a valid Python expression, which can be an input for function eval().
                    '''
                )
            while keys:
                d = {keys.pop():d}

            cfg.update(d)

    return cfg
