import yaml


def get_global_configuration():
    TORCH_CONFIG = "./common_globals/global_configurations.yaml"
    config = None


    with open(TORCH_CONFIG, 'r') as file:
        config = yaml.safe_load(file)

    assert config is not None, "Configuration File missing or is empty."
    return config


if __name__ == '__main__':
    print(get_global_configuration())
