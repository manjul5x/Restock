"""
Data configuration management.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def get_input_file_path(file_key: str) -> Path:
    """
    Get path to an input file by key.
    
    Args:
        file_key: Key from input_files section of config
        
    Returns:
        Path to the input file
        
    Raises:
        KeyError: If file_key not found in config
    """
    config = DataConfig()
    return config.get_input_file_path(file_key)


class DataConfig:
    """
    Load and manage data paths from YAML configuration.
    """
    
    def __init__(self, config_path: str = "data/config/data_config.yaml"):
        """
        Initialize with configuration file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_storage_type(self) -> str:
        """Get storage type from config"""
        return self.config.get('storage', {}).get('type', 'csv')
    
    def get_base_dir(self) -> Path:
        """Get base directory from config"""
        base_dir = self.config.get('paths', {}).get('base_dir', '')
        return Path(base_dir)
    
    def get_product_master_path(self) -> Path:
        """Get product master file path"""
        base_dir = self.get_base_dir()
        filename = self.config.get('paths', {}).get('product_master', '')
        return base_dir / filename
    
    def get_outflow_path(self) -> Path:
        """Get outflow file path"""
        base_dir = self.get_base_dir()
        filename = self.config.get('paths', {}).get('outflow', '')
        return base_dir / filename
    
    def get_output_dir(self) -> Path:
        """Get output directory"""
        output_dir = self.config.get('paths', {}).get('output_dir', 'output')
        return Path(output_dir)
    
    def validate_paths(self) -> bool:
        """Validate that all required paths exist"""
        required_files = [
            self.get_product_master_path(),
            self.get_outflow_path()
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required data file not found: {file_path}")
        
        return True
    
    def get_input_file_path(self, file_key: str) -> Path:
        """
        Get path to an input file by key.
        
        Args:
            file_key: Key from input_files section of config
            
        Returns:
            Path to the input file
            
        Raises:
            KeyError: If file_key not found in config
        """
        input_files = self.config.get('paths', {}).get('input_files', {})
        if file_key not in input_files:
            raise KeyError(f"Input file key '{file_key}' not found in config")
        
        file_path = Path(input_files[file_key])
        
        # If it's a relative path, make it relative to the config file location
        if not file_path.is_absolute():
            file_path = self.config_path.parent.parent.parent / file_path
        
        return file_path 