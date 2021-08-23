import yaml


def config_banner(config, config_name):
    return f'----------------------------------------\n' \
           f'{config_name}\n' \
           f'----------------------------------------\n' \
           f'{yaml.dump(config, default_flow_style=False)}' \
           f'----------------------------------------\n'


def save_config(config, config_filename):
    with open(config_filename, 'w') as f:
        return yaml.dump(config, f, default_flow_style=False)


def load_config(config_filename):
    with open(config_filename, 'r') as f:
        return yaml.safe_load(f)
