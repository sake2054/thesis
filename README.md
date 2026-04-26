# README

`python -m venv .venv`

windows: `.venv\Scripts\activate.bat`

macos: `source .venv/bin/activate`

`pip install pandas numpy scikit-learn lightgbm tensorflow matplotlib seaborn`

`python benchmark_dsl_continuous_auth.py`

## Node/React web demo

All web-service files live under `demo/`.

```bash
cd demo
npm install
ADMIN_PIN=change-me npm run dev
```

For deployment with PM2 and Nginx reverse proxy, see `demo/README.md` and
`demo/docs/nginx-keystroke-demo.conf`.
