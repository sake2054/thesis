# Keystroke Authentication Demo

React + Vite frontend, Node.js + Express API, and SQLite storage for keystroke
collection and browser-side authentication demos.

## Local Run

```bash
cd demo
npm install
ADMIN_PIN=change-me npm run dev
```

Open `http://127.0.0.1:3000`.

## Production Build

```bash
cd demo
npm install
npm run build
ADMIN_PIN='replace-this-pin' npm start
```

The Node app listens on `127.0.0.1:3000` by default and serves both the API and
the built React app.

## PM2

```bash
cd demo
pm2 start ecosystem.config.cjs
pm2 save
```

Set these environment variables in `ecosystem.config.cjs` or your deployment
environment:

- `PORT`: default `3000`
- `DATABASE_PATH`: default `web_demo_data/keystroke_demo.sqlite`
- `ADMIN_PIN`: required for `/admin`
- `CONSENT_VERSION`: consent text version
- `FIXED_PROMPT_TEXT`: assigned fixed text shown behind the input field

## Nginx

Use `docs/nginx-keystroke-demo.conf` as a starting point for a root-domain
reverse proxy.

## Data

The app stores consent, IP address, user agent, device metadata, raw text,
keyboard/input/composition events, extracted features, and model results in
SQLite. Admin CSV exports are available at `/admin` after entering `ADMIN_PIN`.

## Model Artifacts

`public/models/manifest.json` is intentionally optional. When LightGBM or
TensorFlow.js artifacts are absent, the app still collects data and runs the
instant enrollment baseline.
