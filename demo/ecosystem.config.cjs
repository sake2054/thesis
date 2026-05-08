module.exports = {
    apps: [
        {
            name: "keystroke-auth-demo",
            script: "server/index.js",
            cwd: __dirname,
            env: {
                NODE_ENV: "production",
                HOST: "0.0.0.0",
                PORT: "45670",
                DATABASE_PATH: "web_demo_data/keystroke_demo.sqlite",
                CONSENT_VERSION: "2026-05-08",
                FIXED_PROMPT_TEXT:
                    "Type the assigned research phrase exactly as shown.",
                ADMIN_PIN: "qazxsw22@",
            },
        },
    ],
};
