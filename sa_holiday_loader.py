# sa_holiday_loader.py

import holidays
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sa_holidays(start_year: int = 2020, end_year: int = 2026) -> set:
    """
    Return a set of public holidays in South Africa from start_year to end_year (inclusive).

    Returns:
    - set of datetime.date objects for fast holiday lookup
    """
    logger.info(f"📅 Loading SA holidays from {start_year} to {end_year}")
    try:
        sa_holiday_dict = holidays.SouthAfrica(years=range(start_year, end_year + 1))
        return set(sa_holiday_dict.keys())
    except Exception as e:
        logger.error(f"🚫 Failed to load SA holidays: {e}")
        return set()
