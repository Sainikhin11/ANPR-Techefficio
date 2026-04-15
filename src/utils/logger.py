
import sys
from loguru import logger
import yaml
import os

def setup_logger(config_path="config/config.yaml", log_file_override=None):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback if config not found immediately
        config = {"logging": {"level": "INFO", "file_path": "logs/anpr_system.log", "rotation": "10 MB"}}

    log_level = config.get("logging", {}).get("level", "INFO")
    log_file = log_file_override or config.get("logging", {}).get("file_path", "logs/anpr_system.log")
    rotation = config.get("logging", {}).get("rotation", "10 MB")

    # Remove default handler
    logger.remove()

    # Console Handler
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # File Handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger.add(
        log_file,
        rotation=rotation,
        retention="7 days",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    return logger

# Singleton instance
log = setup_logger()
