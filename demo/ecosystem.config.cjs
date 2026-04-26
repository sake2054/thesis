module.exports = {
  apps: [
    {
      name: "keystroke-auth-demo",
      script: "server/index.js",
      cwd: __dirname,
      env: {
        NODE_ENV: "production",
        PORT: "3000",
        DATABASE_PATH: "web_demo_data/keystroke_demo.sqlite",
        CONSENT_VERSION: "2026-04-26",
        FIXED_PROMPT_TEXT: "Type the assigned research phrase exactly as shown.",
        ADMIN_PIN: "change-me"
      }
    }
  ]
};
