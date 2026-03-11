# AbbiTower AI Command Dashboard

## Streamlit dashboard

Run the AbbiTower AI command dashboard with demo/preview disabled by default:

```bash
pip install -r requirements.txt
export MULTIPLEX_ENABLE_DEMO_MODE=false
export MULTIPLEX_ENABLE_PREVIEW_MODE=false
streamlit run streamlit_app.py
```

To enable demo access (only after the production build is ready), explicitly
set both preview + demo to true:

```bash
export MULTIPLEX_ENABLE_PREVIEW_MODE=true
export MULTIPLEX_ENABLE_DEMO_MODE=true
```

Login using **User ID** `1` and **Password** `qwerty` for the demo gate when
demo mode is enabled.
For production usage, set `MULTIPLEX_AUTH_USER` and `MULTIPLEX_AUTH_PASSWORD`
or configure Streamlit `secrets.toml` to replace the demo credentials, and set
`MULTIPLEX_ENABLE_DEMO_MODE=false` (and leave preview disabled) to disable demo
access.

> ⚠️ **Security warning:** Demo credentials are intentionally weak and must not
> be used in any publicly accessible deployment.
