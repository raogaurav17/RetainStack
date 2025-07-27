from logger.logger import get_logger

logger = get_logger(__name__)  # Logger name will be 'test_logger'

def divide(a, b):
    logger.debug(f"Starting division: {a} / {b}")
    try:
        result = a / b
        logger.info(f"Division successful: {result}")
        return result
    except ZeroDivisionError as e:
        logger.error("Attempted division by zero")
        logger.exception("Exception occurred in divide function")
        return None

def main():
    logger.info("Logger test started")
    divide(10, 2)   # Should log debug and info
    divide(5, 0)    # Should log error and exception
    logger.warning("This is a warning")
    logger.critical("This is a critical message")
    logger.info("Logger test complete")

if __name__ == "__main__":
    main()
