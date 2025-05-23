"""
Configuration loading utilities for newspaper analysis project.
"""

import os
import re
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file and substitute environment variables.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration with environment variables substituted
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load environment variables from .env file if it exists
    project_root = Path(config_path).parent.parent
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file)
    
    # Read the YAML file as a string first
    with open(config_path, 'r', encoding='utf-8') as file:
        yaml_str = file.read()
    
    # Replace environment variables in the format ${VAR_NAME}
    pattern = r'\${([^}]+)}'
    
    def replace_env_var(match):
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            print(f"Warning: Environment variable '{var_name}' not found")
            return match.group(0)  # Return the original placeholder if not found
        return env_value
    
    # Substitute environment variables
    yaml_str = re.sub(pattern, replace_env_var, yaml_str)
    
    # Parse the YAML with substituted values
    config = yaml.safe_load(yaml_str)
    
    return config


def get_config_section(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    """
    Get a specific section from the configuration.
    
    Args:
        config: Configuration dictionary
        section: Section name to retrieve
        
    Returns:
        Dictionary containing the requested section
    """
    if section not in config:
        raise KeyError(f"Section '{section}' not found in configuration")
    
    return config[section]


def resolve_paths(config: Dict[str, Any], base_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolve relative paths in configuration.
    
    Args:
        config: Configuration dictionary
        base_dir: Base directory for resolving relative paths
        
    Returns:
        Configuration with resolved paths
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Deep copy the config to avoid modifying the original
    resolved_config = {}
    
    for key, value in config.items():
        if isinstance(value, dict):
            resolved_config[key] = resolve_paths(value, base_dir)
        elif isinstance(value, str) and (value.startswith('./') or value.startswith('../')):
            resolved_config[key] = os.path.normpath(os.path.join(base_dir, value))
        else:
            resolved_config[key] = value
    
    return resolved_config
