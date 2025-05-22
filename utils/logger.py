# utils/logger.py

import logging
import sys # For stdout if config is not available initially
try:
    from config import DEFAULT_LOGGER_NAME, LOG_FILE_PATH, LOG_LEVEL_CONSOLE, LOG_LEVEL_FILE
except ImportError:
    # Fallback defaults if config.py is not found during early import or testing
    print("Warning: config.py not found or not accessible, using fallback logger settings.", file=sys.stderr)
    DEFAULT_LOGGER_NAME = "rag_content_scraper_fallback"
    LOG_FILE_PATH = None
    LOG_LEVEL_CONSOLE = "INFO"
    LOG_LEVEL_FILE = "DEBUG"


def setup_logger(name=None, log_file=None, console_level_str=None, file_level_str=None):
    logger_name = name if name else DEFAULT_LOGGER_NAME
    logger = logging.getLogger(logger_name)

    # Prevent multiple handlers if logger is already configured
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG) # Set lowest level at logger, handlers control effective level

    actual_log_file = log_file if log_file is not None else LOG_FILE_PATH
    actual_console_level_str = console_level_str if console_level_str else LOG_LEVEL_CONSOLE
    actual_file_level_str = file_level_str if file_level_str else LOG_LEVEL_FILE

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler(sys.stdout) # Use sys.stdout
    try:
        ch_level = getattr(logging, actual_console_level_str.upper(), logging.INFO)
    except AttributeError:
        ch_level = logging.INFO
        print(f"Warning: Invalid console log level '{actual_console_level_str}' in config. Using INFO.", file=sys.stderr)
    ch.setLevel(ch_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (if path is provided)
    if actual_log_file:
        try:
            fh = logging.FileHandler(actual_log_file, encoding='utf-8')
            try:
                fh_level = getattr(logging, actual_file_level_str.upper(), logging.DEBUG)
            except AttributeError:
                fh_level = logging.DEBUG
                print(f"Warning: Invalid file log level '{actual_file_level_str}' in config. Using DEBUG.", file=sys.stderr)
            fh.setLevel(fh_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            # logger.info(f"File logging enabled: {actual_log_file} at level {actual_file_level_str.upper()}")
        except Exception as e:
            logger.error(f"Failed to configure file logger for {actual_log_file}: {e}", exc_info=False)
            # Fallback to console to ensure error is visible
            print(f"ERROR: Failed to configure file logger for {actual_log_file}: {e}", file=sys.stderr)


    # Initial log to confirm setup (optional)
    # logger.debug(f"Logger '{logger_name}' configured. Console: {actual_console_level_str}, File: {actual_log_file} ({actual_file_level_str})")
    return logger