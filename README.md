# popapapipipapa

## Streamlit dashboard

Run the mobile-inspired admin dashboard with:

```bash
pip install -r requirements.txt
export MULTIPLEX_ENABLE_DEMO_MODE=true
streamlit run streamlit_app.py
```

Login using **User ID** `1` and **Password** `qwerty` for the demo gate when
demo mode is enabled.
For production usage, set `MULTIPLEX_AUTH_USER` and `MULTIPLEX_AUTH_PASSWORD`
or configure Streamlit `secrets.toml` to replace the demo credentials, and set
`MULTIPLEX_ENABLE_DEMO_MODE=false` to disable demo access.

> ⚠️ **Security warning:** Demo credentials are intentionally weak and must not
> be used in any publicly accessible deployment.
