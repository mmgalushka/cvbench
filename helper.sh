#!/bin/bash

# =============================================================================
# CVBENCH HELPER
# =============================================================================

NC=$(echo "\\033[m")
BOLD=$(echo "\\033[1;39m")
CMD=$(echo "\\033[1;34m")
OPT=$(echo "\\033[0;34m")

action_usage(){
    echo -e "  ______     ______                  _     ";
    echo -e " / ___\\ \\   / / __ )  ___ _ __   ___| |__  ";
    echo -e "| |    \\ \\ / /|  _ \\ / _ \\ '_ \\ / __| '_ \\ ";
    echo -e "| |___  \\ V / | |_) |  __/ | | | (__| | | |";
    echo -e " \\____|  \\_/  |____/ \\___|_| |_|\\___|_| |_|";
    echo -e "Computer Vision Training Sandbox"
    echo -e ""
    echo -e "${BOLD}Dev commands (./helper.sh <name>):${NC}"
    echo -e "  ${CMD}init${NC}                 create .venv and install all dependencies"
    echo -e "  ${CMD}test${OPT} [-m mark] [-c]${NC}  run the test suite (-c adds a coverage summary)"
    echo -e "  ${CMD}release${OPT} [--dry-run]${NC}  preview the next version bump (CI does the real one)"
    echo -e "  ${CMD}docs${NC}                 regenerate the CLI reference block in README.md"
    echo -e ""
    echo -e "  ${CMD}data|train|evaluate|predict|runs|augmentations|serve${NC}  pass through to the CLI"
    echo -e ""
    if [ -x .venv/bin/commands ]; then
        .venv/bin/commands
    else
        echo -e "Run ${CMD}./helper.sh init${NC} first, then ${CMD}./helper.sh <command> --help${NC}"
        echo -e "or ${CMD}commands${NC} inside the container for the full command list."
    fi
}

action_docs(){
    action_activate
    python - <<'EOF'
import re, pathlib
from cvbench.cli.overview import render_markdown, render_quickstart_markdown

readme = pathlib.Path("README.md")
text = readme.read_text()
new = text
changed = []
missing = []
for marker, block in (
    ("QUICKSTART", render_quickstart_markdown()),
    ("CLI REFERENCE", render_markdown()),
):
    pattern = rf"(<!-- BEGIN {marker} -->\n).*?(\n<!-- END {marker} -->)"
    if not re.search(pattern, new, flags=re.DOTALL):
        missing.append(marker)
        continue
    updated = re.sub(pattern, lambda m: m.group(1) + block + m.group(2), new, flags=re.DOTALL)
    if updated != new:
        changed.append(marker)
    new = updated

if missing:
    print(f"markers <!-- BEGIN/END {'/'.join(missing)} --> not found in README.md")
if changed:
    readme.write_text(new)
    print(f"README.md regenerated: {', '.join(changed)}.")
elif not missing:
    print("README.md already up to date.")
EOF
}

action_init(){
    if [ -d .venv ]; then
        rm -rf .venv
    fi

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip -q

    if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
        echo "Apple Silicon detected — installing tensorflow-metal for GPU acceleration"
        pip install -e ".[dev,web,export,gpu-mac]"
    else
        pip install -e ".[dev,web,export]"
    fi
}

action_activate(){
    if [ ! -f .venv/bin/activate ]; then
        echo "Virtual environment not found. Run: ./helper.sh init"
        exit 1
    fi
    source .venv/bin/activate
}

action_data(){
    action_activate
    data "$@"
}

action_train(){
    action_activate
    train "$@"
}

action_evaluate(){
    action_activate
    evaluate "$@"
}

action_predict(){
    action_activate
    predict "$@"
}

action_runs(){
    action_activate
    runs "$@"
}

action_augmentations(){
    action_activate
    augmentations "$@"
}

action_serve(){
    action_activate
    serve "$@"
}

action_test(){
    action_activate
    OPTS=()
    while getopts ":m:c" opt; do
        case $opt in
            m) OPTS+=(-m "$OPTARG") ;;
            c) OPTS+=(--cov=cvbench --cov-report=term-missing) ;;
            \?) echo "Invalid option: -$OPTARG"; exit 1 ;;
        esac
    done
    pytest "${OPTS[@]}"
}

action_release(){
    action_activate
    if [[ "$1" == "--dry-run" ]]; then
        cz bump --dry-run
    else
        echo "Release is handled automatically by the GitHub Actions pipeline."
        echo "To preview what version would be bumped, run: ./helper.sh release --dry-run"
    fi
}

# =============================================================================
# HELPER COMMANDS SELECTOR
# =============================================================================
case $1 in
    init)
        action_init
        ;;
    data)
        action_data ${@:2}
        ;;
    train)
        action_train ${@:2}
        ;;
    evaluate)
        action_evaluate ${@:2}
        ;;
    predict)
        action_predict ${@:2}
        ;;
    runs)
        action_runs ${@:2}
        ;;
    augmentations)
        action_augmentations ${@:2}
        ;;
    serve)
        action_serve ${@:2}
        ;;
    test)
        action_test ${@:2}
        ;;
    release)
        action_release ${@:2}
        ;;
    docs)
        action_docs
        ;;
    *)
        action_usage
        ;;
esac

exit 0
