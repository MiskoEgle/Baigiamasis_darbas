import logging
import os
import codecs
from logging.handlers import RotatingFileHandler
from datetime import datetime
import sys

# Sukurti logų katalogą
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Sukurti logų failo pavadinimą su data
LOG_FILE = os.path.join(LOG_DIR, f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

def setup_logger():
    """
    Sukonfigūruoti loggerį
    """
    try:
        # Sukurti loggerį
        logger = logging.getLogger('app')
        logger.setLevel(logging.DEBUG)
        
        # Sukurti formatterį
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Sukurti failo handlerį
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=10485760,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Sukurti konsolės handlerį
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Pridėti handlerius
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    except Exception as e:
        print(f"Klaida konfigūruojant loggerį: {str(e)}")
        raise

# Sukurti loggerį
logger = setup_logger()

# Logging funkcijos
def log_info(message):
    """
    Įrašyti info pranešimą
    """
    logger.info(message)
    
def log_error(message):
    """
    Įrašyti klaidos pranešimą
    """
    logger.error(message)
    
def log_debug(message):
    """
    Įrašyti debug pranešimą
    """
    logger.debug(message)
    
def log_warning(message):
    """
    Įrašyti įspėjimo pranešimą
    """
    logger.warning(message)
    
def log_critical(message):
    """
    Įrašyti kritinio pranešimo pranešimą
    """
    logger.critical(message)

def get_logger(name):
    """
    Gauti logger'į pagal vardą
    """
    return logging.getLogger(name)

def log_error(logger, error, context=None):
    """
    Užregistruoti klaidą
    """
    try:
        error_msg = f"Klaida: {str(error)}"
        if context:
            error_msg += f" | Kontekstas: {context}"
        logger.error(error_msg)
        
    except Exception as e:
        print(f"Klaida registruojant klaidą: {str(e)}")

def log_info(logger, message, context=None):
    """
    Užregistruoti informacinį pranešimą
    """
    try:
        info_msg = message
        if context:
            info_msg += f" | Kontekstas: {context}"
        logger.info(info_msg)
        
    except Exception as e:
        print(f"Klaida registruojant informaciją: {str(e)}")

def log_warning(logger, message, context=None):
    """
    Užregistruoti įspėjimą
    """
    try:
        warning_msg = message
        if context:
            warning_msg += f" | Kontekstas: {context}"
        logger.warning(warning_msg)
        
    except Exception as e:
        print(f"Klaida registruojant įspėjimą: {str(e)}")

def log_debug(logger, message, context=None):
    """
    Užregistruoti debug pranešimą
    """
    try:
        debug_msg = message
        if context:
            debug_msg += f" | Kontekstas: {context}"
        logger.debug(debug_msg)
        
    except Exception as e:
        print(f"Klaida registruojant debug pranešimą: {str(e)}") 