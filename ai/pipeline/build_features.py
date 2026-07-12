from load_features import save_feature_vectors
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


def build_feature_vectors():

    # -----------------------------
    # Load Data
    # -----------------------------
    leads = get_leads(source="mock")
    companies = get_companies(source="mock")
    activities = get_activities(source="mock")
    emails = get_emails(source="mock")

    

    merged = leads.merge(companies, on="company_id")

    

    feature_vectors = []

    # -----------------------------
    # Build Features
    # -----------------------------
    for _, lead in merged.iterrows():

        lead_id = lead["lead_id"]

        lead_activities = activities[
            activities["lead_id"] == lead_id
        ]

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

        feature_vector = {

            "lead_id": lead_id,

            "feature_version": "v1",

            "generated_at": str(pd.Timestamp.now()),

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

    # RETURN AFTER LOOP
    return pd.DataFrame(feature_vectors)


if __name__ == "__main__":

    features_df = build_feature_vectors()
    save_feature_vectors(features_df)

    print("\nGenerated Feature Vectors\n")

    print(features_df)

    features_df.to_csv(
        "ai/mock_data/feature_vectors.csv",
        index=False
    )

    features_df.to_json(
        "ai/mock_data/feature_vectors.json",
        orient="records",
        indent=4
    )

    print("\nFeature vectors saved successfully.")