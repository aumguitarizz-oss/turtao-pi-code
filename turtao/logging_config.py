import logging
import logging.handlers
import sys
from pathlib import Path

LOG_DIR = Path("/var/log/turtao")
LOG_FILE = LOG_DIR / "turtao.log"
FALLBACK_DIR = Path(__file__).resolve().parent.parent / "logs"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3
LOG_FORMAT = "%(asctime)s [%(threadName)s] %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(debug: bool = False) -> None:
    log_dir = LOG_DIR
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        log_dir = FALLBACK_DIR
        log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "turtao.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(console_handler)

    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("flask").setLevel(logging.WARNING)
