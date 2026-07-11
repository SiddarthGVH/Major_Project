import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MOCK_DIR = os.path.join(BASE_DIR, "mock_data")


def get_leads():
    return pd.read_csv(os.path.join(MOCK_DIR, "leads.csv"))


def get_companies():
    return pd.read_csv(os.path.join(MOCK_DIR, "companies.csv"))


def get_activities():
    return pd.read_csv(os.path.join(MOCK_DIR, "activities.csv"))


def get_emails():
    return pd.read_csv(os.path.join(MOCK_DIR, "emails.csv"))