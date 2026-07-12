import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MOCK_DIR = os.path.join(BASE_DIR, "mock_data")


def get_leads(source="mock"):
    if source == "mock":
        return pd.read_csv(os.path.join(MOCK_DIR, "leads.csv"))
    else:
        raise NotImplementedError("Database source not implemented yet")


def get_companies(source="mock"):
    if source == "mock":
        return pd.read_csv(os.path.join(MOCK_DIR, "companies.csv"))
    else:
        raise NotImplementedError("Database source not implemented yet")


def get_activities(source="mock"):
    if source == "mock":
        return pd.read_csv(os.path.join(MOCK_DIR, "activities.csv"))
    else:
        raise NotImplementedError("Database source not implemented yet")


def get_emails(source="mock"):
    if source == "mock":
        return pd.read_csv(os.path.join(MOCK_DIR, "emails.csv"))
    else:
        raise NotImplementedError("Database source not implemented yet")