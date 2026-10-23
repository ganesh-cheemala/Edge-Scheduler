# Hosted deployment

## Target

This project uses Streamlit, so the simplest managed host is Streamlit Community Cloud. It deploys from a GitHub repository. The hosted package is designed to run the model-generated workload without uploading the downloaded source archives, consolidated task tables, or generated research results.

## Publish the hosted package

1. Create a GitHub repository for the hosted app. Choose whether it should be public or private.
2. Upload the contents of the `hosted deployment` folder to the repository root.
3. In Streamlit Community Cloud, connect the GitHub account and create an app from that repository. Set the entry point to `app.py` and choose Python 3.12 in the advanced settings.
4. Deploy, open the provided URL, and run a policy comparison from the sidebar.

The included `.gitignore` excludes local raw data, processed data, and generated simulation results. Consequently, the hosted app defaults to model-generated task archetypes. The reference-informed option appears only when its consolidated task table is available beside the app. Local dataset preparation remains available in the full project folder.

## Hosting notes

- No secrets or external services are required by the current app.
- Generated files on the hosted service are temporary and may be lost when the app restarts. Download reports from the interface if they need to be retained.
- Long multi-seed and scalability runs may take longer on shared hosting than on a local computer. Start with the default single policy comparison before running those studies.
- All latency, energy, and scheduling performance values are software-model estimates. They are not measurements from a physical IoT deployment.

## Current deployment dependency versions

The hosted package pins the Streamlit, pandas, and Plotly versions used during preparation. If those versions change, run the application locally and update the pins deliberately.
