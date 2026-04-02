import yaml
import os


def load_config(config_path="config.yaml"):
    """
    加载 YAML 配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_config_value(config, *keys, default=None):
    """
    从嵌套字典中安全获取配置值
    
    Args:
        config: 配置字典
        *keys: 配置键路径
        default: 默认值
        
    Returns:
        配置值或默认值
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value
