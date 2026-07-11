from data_source import (
    get_leads,
    get_companies,
    get_activities,
    get_emails,
)

from features import (
    company_size_band,
    source_quality,
    engagement_score,
    engagement_level,
    reply_rate,
    reply_level,
    recency_days,
)

import pandas as pd


# -----------------------------
# Load data
# -----------------------------

leads = get_leads()
companies = get_companies()
activities = get_activities()
emails = get_emails()

# Merge leads with company details
merged = leads.merge(companies, on="company_id")


# Store all feature vectors
feature_vectors = []


# -----------------------------
# Process each lead
# -----------------------------

for _, lead in merged.iterrows():

    lead_id = lead["lead_id"]

    # Activities for current lead
    lead_activities = activities[
        activities["lead_id"] == lead_id
    ]

    # Emails for current lead
    lead_emails = emails[
        emails["lead_id"] == lead_id
    ]


    company_band = company_size_band(
        lead["size"]
    )



    source_score = source_quality(
        lead["source"]
    )

  

    engagement = engagement_score(
        lead_activities
    )

    engagement_status = engagement_level(
        engagement
    )

    reply_score = reply_rate(
    lead_emails
    )

    reply_status = reply_level(
    reply_score
    )

    recency = recency_days(
    lead_activities
   )

    # -------------------------
    # Feature Vector
    # -------------------------

    feature_vector = {

        "lead_id": lead_id,

        "company": lead["name"],

        "company_band": company_band,

        "source": lead["source"],

        "source_score": source_score,

        "engagement_score": engagement,

        "engagement_level": engagement_status,

        "reply_rate": reply_score,

        "reply_level": reply_status,

        "recency_days": recency

    }

    feature_vectors.append(feature_vector)


# ----------------------------------
# Convert to DataFrame
# ----------------------------------

features_df = pd.DataFrame(feature_vectors)

print("\nGenerated Feature Vectors:\n")

print(features_df)


# ----------------------------------
# Save Output
# ----------------------------------

features_df.to_csv(
    "ai/mock_data/feature_vectors.csv",
    index=False
)

print("\nFeature vectors saved successfully.")

