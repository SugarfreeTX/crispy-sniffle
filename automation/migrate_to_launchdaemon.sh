#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/khemra/dev/crispy-sniffle"
SRC_PLIST="${REPO_ROOT}/automation/com.grok.portfolio.daily.daemon.plist"
DST_PLIST="/Library/LaunchDaemons/com.grok.portfolio.daily.daemon.plist"
USER_NAME="khemra"
USER_GROUP="staff"
USER_UID="$(id -u "${USER_NAME}")"
AGENT_LABEL="com.grok.portfolio.daily"
DAEMON_LABEL="com.grok.portfolio.daily.daemon"

echo "[1/6] Validating source plist"
plutil -lint "${SRC_PLIST}" >/dev/null

echo "[2/6] Installing LaunchDaemon plist"
cp "${SRC_PLIST}" "${DST_PLIST}"
chown root:wheel "${DST_PLIST}"
chmod 644 "${DST_PLIST}"

echo "[3/6] Ensuring log files exist and are user-writable"
touch "${REPO_ROOT}/trading_launchd.out.log" "${REPO_ROOT}/trading_launchd.err.log"
chown "${USER_NAME}:${USER_GROUP}" "${REPO_ROOT}/trading_launchd.out.log" "${REPO_ROOT}/trading_launchd.err.log"
chmod 664 "${REPO_ROOT}/trading_launchd.out.log" "${REPO_ROOT}/trading_launchd.err.log"

echo "[4/6] Disabling user LaunchAgent"
launchctl bootout "gui/${USER_UID}/${AGENT_LABEL}" 2>/dev/null || true
launchctl disable "gui/${USER_UID}/${AGENT_LABEL}" 2>/dev/null || true

echo "[5/6] Enabling and bootstrapping system LaunchDaemon"
launchctl bootout "system/${DAEMON_LABEL}" 2>/dev/null || true
launchctl enable "system/${DAEMON_LABEL}" 2>/dev/null || true
launchctl bootstrap system "${DST_PLIST}"

echo "[6/6] Verifying daemon status"
launchctl print "system/${DAEMON_LABEL}" | grep -E "state =|path =|last exit code|runs =" || true

echo "Done. The system daemon ${DAEMON_LABEL} is installed from ${DST_PLIST}."
