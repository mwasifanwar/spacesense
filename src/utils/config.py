import yaml

class Config:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.data = yaml.safe_load(file)
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.data
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default

config = Config()