# Keystroke Authentication Research Demo

React + Vite frontend, Node.js + Express API, and SQLite storage for browser
keystroke dynamics data collection and browser-side authentication demos.

## Local Run

```bash
cd demo
npm install
ADMIN_PIN=change-me npm run dev
```

Open `http://127.0.0.1:3000`.

Researcher-issued links can set the experimental condition:

```text
/?participantCode=P001&role=genuine&sessionNo=1
/?participantCode=P002&role=imposter&targetParticipantCode=P001&sessionNo=1
/?participantCode=P001&role=genuine&sessionNo=1&inputMode=free&monitoring=1
```

`?devControls=1` shows role/input/model debugging controls. General
participant UI only displays the assigned role and condition.

## Production Build

```bash
cd demo
npm install
npm run build
ADMIN_PIN='replace-this-pin' npm start
```

The Node app listens on `127.0.0.1:3000` by default and serves both the API and
the built React app.

## Data Collected

This research demo intentionally stores raw input data for analysis. Consent
text and deployment instructions must make this explicit.

- Raw text typed by the participant.
- Key value and code for keyboard events when `STORE_KEY_VALUE=true`.
- `keydown`, `keyup`, `beforeinput`, `input`, `composition*`, and `paste` events.
- Input event payload metadata such as timestamps, repeat/composition flags,
  input type, and current value length.
- IP address when `STORE_IP_ADDRESS=true`.
- User agent, viewport, screen, browser/navigator metadata, language, timezone,
  touch support, and detected device class.
- Participant code and SHA-256 participant code hash. Raw participant code is
  stored only when `STORE_PARTICIPANT_CODE=true`.
- Sessions, attempts, prompt ids, role labels, target participant codes,
  extracted features, quality summaries, model results, monitoring windows, and
  calibration thresholds.

Participants must be told not to type names, phone numbers, addresses, account
ids, passwords, or other sensitive personal information.

## Attempt Lifecycle

Raw events are flushed continuously while typing. Final feature extraction and
model result storage happen only when the participant clicks **Submit attempt**.
Attempts use these statuses:

- `in_progress`
- `submitted`
- `cancelled`
- `excluded`

Paste policy defaults:

- fixed mode paste: `quality_status='excluded'`
- free mode paste: `quality_status='low_quality'`

The policies can be changed with `PASTE_POLICY_FIXED` and `PASTE_POLICY_FREE`.

## Prompt Sets

Prompt sets are JSON-file based. The default file is
`server/prompts.default.json`, controlled by `PROMPT_SET_PATH`.

Schema:

```json
{
  "promptSets": [
    {
      "id": "pilot_fixed_ko_v1",
      "label": "Korean fixed prompts v1",
      "inputMode": "fixed",
      "language": "ko",
      "prompts": [
        { "id": "ko_fixed_001", "text": "Prompt text", "minChars": 40 }
      ]
    }
  ]
}
```

Fixed mode shows prompts in trial order and stores `prompt_set_id`,
`prompt_id`, `prompt_text`, raw text, exact match, and edit distance. Free mode
can show a randomized writing suggestion; the app stores `suggestionShown` and
`suggestionId` while preserving the participant's raw text.

## Model Artifact Manifest

`public/models/manifest.json` supports device/inputMode-specific models:

```json
{
  "version": "0.2.0",
  "artifactAvailable": true,
  "models": {
    "desktop": {
      "fixed": {
        "lightgbm": {
          "path": "/models/lightgbm/desktop_fixed_model.json",
          "threshold": 0.35,
          "featureCount": 15,
          "mean": [],
          "scale": [],
          "featureNames": []
        },
        "cnn1d": {
          "modelJson": "/models/cnn1d/desktop_fixed/model.json",
          "threshold": 0.42,
          "inputShape": [15, 1],
          "mean": [],
          "scale": []
        }
      },
      "free": {}
    },
    "mobile": { "fixed": {}, "free": {} },
    "tablet": { "fixed": {}, "free": {} },
    "default": { "fixed": {}, "free": {} }
  }
}
```

Selection fallback order:

1. `deviceClass + inputMode`
2. `default + inputMode`
3. `desktop + inputMode`
4. any available model

LightGBM artifacts use Python `Booster.dump_model()` JSON. The browser runtime
handles `default_left` and missing values. TensorFlow.js 1D-CNN artifacts are
loaded with `tf.loadLayersModel()` and reshaped according to `inputShape`.

## Web Worker Inference

Optional LightGBM and TFJS 1D-CNN inference runs in `src/inference.worker.js`
through `src/inferenceClient.js`. The worker loads the manifest, caches model
artifacts, predicts, and returns worker-side `inferenceTimeMs`. The main thread
measures round-trip `uiBlockingTimeMs`. If workers fail, the app falls back to
the same runtime on the main thread and continues collecting data.

## Continuous Monitoring

Collection mode and monitoring mode are separate. Default collection mode runs
analysis on submit. Monitoring mode is enabled with `?monitoring=1` or
`CONTINUOUS_MONITORING_ENABLED=true`. It stores sliding windows in
`monitoring_windows`.

Defaults:

- `MONITORING_WINDOW_CHARS=120`
- `MONITORING_STEP_CHARS=60`

## Calibration

The `model_calibrations` table stores participant/model/device/input thresholds.
Prediction threshold priority:

1. participant-specific calibration threshold
2. device/inputMode/model calibration
3. manifest threshold
4. `0.5`

Admin endpoints:

- `GET /api/admin/calibrations`
- `POST /api/admin/calibrate`

The default automatic method uses the lower
`CALIBRATION_GENUINE_QUANTILE=0.05` percentile of genuine enrollment scores.

## Admin Exports

All admin exports require `x-admin-pin`.

Raw table exports:

- `/api/admin/export/participants.csv`
- `/api/admin/export/sessions.csv`
- `/api/admin/export/attempts.csv`
- `/api/admin/export/events.csv`
- `/api/admin/export/features.csv`
- `/api/admin/export/results.csv`
- `/api/admin/export/monitoring_windows.csv`
- `/api/admin/export/model_calibrations.csv`

Analysis-ready exports:

- `/api/admin/export/attempt_features.csv`
- `/api/admin/export/event_pairs.csv`
- `/api/admin/export/model_results.csv`
- `/api/admin/export/quality_summary.csv`
- `/api/admin/export/monitoring_windows.csv`

CSV cells are escaped server-side.

## Environment Variables

- `ADMIN_PIN`: admin console/export PIN.
- `CONSENT_VERSION`: consent text version shown to participants.
- `DATABASE_PATH`: SQLite path. Default `web_demo_data/keystroke_demo.sqlite`.
- `PORT`: default `3000`.
- `HOST`: default `127.0.0.1`.
- `STORE_RAW_TEXT`: default `true`.
- `STORE_KEY_VALUE`: default `true`.
- `STORE_IP_ADDRESS`: default `true`.
- `STORE_PARTICIPANT_CODE`: default `true`.
- `SHOW_DEV_CONTROLS`: default `false`.
- `CONTINUOUS_MONITORING_ENABLED`: default `false`.
- `MONITORING_WINDOW_CHARS`: default `120`.
- `MONITORING_STEP_CHARS`: default `60`.
- `PASTE_POLICY_FIXED`: default `excluded`.
- `PASTE_POLICY_FREE`: default `low_quality`.
- `CALIBRATION_GENUINE_QUANTILE`: default `0.05`.
- `PROMPT_SET_PATH`: default `server/prompts.default.json`.

## Files Added

- `server/prompts.default.json`: default fixed prompts and free writing suggestions.
- `src/inference.worker.js`: Web Worker inference message handler.
- `src/inferenceClient.js`: worker client and main-thread fallback.

## PM2

```bash
cd demo
pm2 start ecosystem.config.cjs
pm2 save
```

Set environment variables in `ecosystem.config.cjs` or your deployment
environment.

## Nginx

Use `docs/nginx-keystroke-demo.conf` as a starting point for a root-domain
reverse proxy.

## Pilot Deployment Checklist

- Set a non-default `ADMIN_PIN`.
- Confirm consent text and `CONSENT_VERSION`.
- Confirm prompt sets and researcher-issued participant links.
- Confirm whether raw participant codes should be stored.
- Keep `STORE_RAW_TEXT=true` and `STORE_KEY_VALUE=true` for this study design,
  and make that clear to participants.
- Add LightGBM/TFJS artifacts to `public/models` and update the manifest.
- Run `npm run build` and a smoke test before deployment.
- Export `attempt_features.csv`, `event_pairs.csv`, and
  `quality_summary.csv` after each pilot batch.
