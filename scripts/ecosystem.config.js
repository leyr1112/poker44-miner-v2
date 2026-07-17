// pm2 process definition for the Poker44 v2 miner.
// SECURITY: no wallet, hotkey or port is hardcoded here (this file is committed).
// All operational config comes from `.env` (gitignored). Copy `.env.example`
// -> `.env` and fill it in before starting.
const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

const ROOT = path.resolve(__dirname, "..");
const PY = process.env.POKER44_PYTHON || `${ROOT}/.venv/bin/python`;

function loadEnv(p) {
  const out = {};
  try {
    for (const raw of fs.readFileSync(p, "utf8").split("\n")) {
      const line = raw.trim();
      if (!line || line.startsWith("#")) continue;
      const i = line.indexOf("=");
      if (i > 0) out[line.slice(0, i).trim()] = line.slice(i + 1).trim().replace(/^["']|["']$/g, "");
    }
  } catch (e) {}
  return out;
}
const E = { ...loadEnv(`${ROOT}/.env`), ...process.env };

const WALLET = E.POKER44_WALLET_NAME;
const HOTKEY = E.POKER44_WALLET_HOTKEY;
const NETUID = E.POKER44_NETUID || "126";
const PORT = E.POKER44_AXON_PORT || "8091";
if (!WALLET || !HOTKEY) {
  throw new Error("missing POKER44_WALLET_NAME / POKER44_WALLET_HOTKEY — create .env from .env.example");
}

let COMMIT = "";
try { COMMIT = execSync(`git -C ${ROOT} rev-parse HEAD`).toString().trim(); } catch (e) {}

module.exports = {
  apps: [
    {
      name: "poker44_miner_v2",
      script: `${ROOT}/neurons/miner.py`,
      interpreter: PY,
      cwd: ROOT,
      args: [
        "--netuid", NETUID,
        "--wallet.name", WALLET,
        "--wallet.hotkey", HOTKEY,
        "--subtensor.network", "finney",
        "--axon.port", PORT,
        "--logging.debug",
        "--blacklist.force_validator_permit",
      ].join(" "),
      env: { ...E, POKER44_MODEL_REPO_COMMIT: COMMIT },
      autorestart: true,
      max_restarts: 20,
      min_uptime: "30s",
      restart_delay: 5000,
      kill_timeout: 10000,
    },
  ],
};
