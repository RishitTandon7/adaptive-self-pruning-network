import logging
import sys
import os

def setup_logger(name="self_pruning_network", log_file="training.log", level=logging.INFO):
    """Sets up a standardized logger for the project."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)) or '.', exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
    return logger
