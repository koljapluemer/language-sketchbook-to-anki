dir := justfile_directory()
apps := env("HOME") / ".local/share/applications"
icons := env("HOME") / ".local/share/icons/hicolor/scalable/apps"
# the app's port (as notebook_anki/server.py); whatever listens there is the running server
port := "8642"

# start the app (or open the browser if it's already running)
run:
    uv run --project {{dir}} python -m notebook_anki

# the server with auto-reload on :8642
dev:
    uv run --project {{dir}} uvicorn notebook_anki.server:app --reload --timeout-graceful-shutdown 2 --port {{port}} --app-dir {{dir}}

# install as desktop app (Fedora/Ubuntu, GNOME); restarts the server if running
reinstall:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync --project {{dir}}
    mkdir -p {{icons}} {{apps}}
    cp {{dir}}/notebook_anki/static/icon.svg {{icons}}/language-sketchbook.svg
    sed -e "s|/ABS/PATH/TO/language-sketchbook-to-anki|{{dir}}|g" -e "s|^Exec=uv |Exec=$(command -v uv) |" \
        {{dir}}/language-sketchbook.desktop > {{apps}}/language-sketchbook.desktop
    update-desktop-database {{apps}} 2>/dev/null || true
    gtk-update-icon-cache -q ~/.local/share/icons/hicolor 2>/dev/null || true
    if fuser -s {{port}}/tcp 2>/dev/null; then
        fuser -s -k -TERM {{port}}/tcp
        while fuser -s {{port}}/tcp; do sleep 0.2; done
        setsid -f uv run --project {{dir}} python -m notebook_anki >/dev/null 2>&1
        echo "restarted"
    fi
