"""Paths and settings shared by every module and notebook."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TELCO_FILE = RAW_DIR / "Telco_customer_churn.xlsx"

load_dotenv(PROJECT_ROOT / ".env")
# The Olist database from Project 1. Only read, never written.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://localhost:5432/olist")

# One fixed seed everywhere, so every run (splits, K-Means, Random Forest) gives the same numbers.
RANDOM_STATE = 42

# Olist data is complete only up to Aug 2018 (see Project 1), so RFM is measured
# "as of" 1 Sep 2018, using orders before that date.
RFM_SNAPSHOT_DATE = "2018-09-01"

# Chart colours (same as the other projects): blue = normal / stays, orange = churn / attention.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#8b8a85"
