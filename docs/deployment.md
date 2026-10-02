# Deployment

DataGuard needs a running Python process. GitHub Pages cannot serve its Streamlit
backend. The repository requires no secrets or paid services.

For Streamlit Community Cloud, use this configuration:

- Repository: the published DataGuard repository.
- Branch: `main`.
- Main file: `app/streamlit_app.py`.
- Python: 3.12.
- Dependencies: root `requirements.txt` installs the local package.

Streamlit discovers `.streamlit/config.toml` at the repository root. Upload size
is bounded and usage statistics are disabled. Account creation or GitHub OAuth
may need account-owner interaction. No live demo URL is claimed unless a
deployment actually succeeds.

Official guidance: [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud)
and [deployment setup](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).

On a hosted deployment, uploaded data is transmitted to and analyzed in that
server's memory. Do not describe this as processing only in the visitor's browser.
Public demos should be used with non-sensitive data. A production deployment
would additionally need authentication, resource quotas, access policy, and
an appropriate operational security review.
