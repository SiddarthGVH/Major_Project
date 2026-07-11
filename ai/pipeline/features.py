import pandas as pd
def company_size_band(size):

    if size < 50:
        return "Small"

    elif size < 500:
        return "Medium"

    elif size < 2000:
        return "Large"

    return "Enterprise"


def source_quality(source):

    mapping = {

        "Referral":100,

        "Website":85,

        "LinkedIn":70,

        "Webinar":75,

        "Event":65,

        "Cold Email":40

    }

    return mapping.get(source,50)


def engagement_score(activities):

    score = 0

    for _, row in activities.iterrows():

        if row["type"] == "email" and row["status"] == "sent":
            score += 5

        elif row["type"] == "email" and row["status"] == "replied":
            score += 15

        elif row["type"] == "call":
            score += 10

        elif row["type"] == "meeting":
            score += 20

        elif row["type"] == "proposal":
            score += 25

    return score


def engagement_level(score):

    if score >= 50:
        return "HIGH"

    elif score >= 25:
        return "MEDIUM"

    return "LOW"

def reply_rate(email_df):

    if email_df.empty:
        return 0

    total_sent = len(email_df)

    total_replied = len(
        email_df[email_df["replied"] == "Yes"]
    )

    return round((total_replied / total_sent) * 100, 2)

def reply_level(rate):

    if rate >= 70:
        return "FAST"

    elif rate >= 40:
        return "MEDIUM"

    elif rate > 0:
        return "SLOW"

    return "NO RESPONSE"

from datetime import datetime

def recency_days(activity_df):

    if activity_df.empty:
        return 999

    latest = pd.to_datetime(
        activity_df["created_at"]
    ).max()

    today = pd.Timestamp.today()

    return (today - latest).days