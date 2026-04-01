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
    echo -e "${BOLD}Setup Commands:${NC}"
    echo -e "  ${CMD}init${NC}                         create .venv and install all dependencies;"
    echo -e ""
    echo -e "${BOLD}Data Commands:${NC}"
    echo -e "  ${CMD}generate${OPT} [opts]${NC}             generate synthetic shapes dataset;"
    echo -e "    ${OPT}--output <dir>${NC}             destination directory (default: data/);"
    echo -e "    ${OPT}--train/--val/--test <N>${NC}   images per class per split;"
    echo -e "    ${OPT}--image-size <N>${NC}           image size in pixels (default: 64);"
    echo -e "    ${OPT}--overwrite${NC}                replace existing output directory;"
    echo -e ""
    echo -e "${BOLD}Training Commands:${NC}"
    echo -e "  ${CMD}train${OPT} <data_dir> [opts]${NC}      run training;"
    echo -e "    ${OPT}--output <dir>${NC}             experiment output directory;"
    echo -e "    ${OPT}--from <exp_dir>${NC}           load config from existing experiment;"
    echo -e "    ${OPT}--backbone <name>${NC}          backbone name (default: efficientnet_b0);"
    echo -e "    ${OPT}--epochs <N>${NC}               number of training epochs;"
    echo -e "    ${OPT}--lr <float>${NC}               learning rate;"
    echo -e "    ${OPT}--batch-size <N>${NC}           batch size;"
    echo -e "    ${OPT}--aug-preset <name>${NC}        augmentation preset;"
    echo -e "    ${OPT}--resume <checkpoint>${NC}      resume from a checkpoint;"
    echo -e "  ${CMD}evaluate${OPT} <run_dir> [opts]${NC}   evaluate a trained model;"
    echo -e "    ${OPT}--split val|test${NC}           dataset split to evaluate on;"
    echo -e "    ${OPT}--output-dir <path>${NC}        where to write eval outputs;"
    echo -e "  ${CMD}predict${OPT} [opts]${NC}             run inference on image(s);"
    echo -e "    ${OPT}--checkpoint <path>${NC}        path to .keras checkpoint;"
    echo -e "    ${OPT}--input <path>${NC}             image file or folder;"
    echo -e ""
    echo -e "${BOLD}Experiment Commands:${NC}"
    echo -e "  ${CMD}runs list${OPT} [dir] [--sort val_accuracy|date|backbone]${NC}"
    echo -e "  ${CMD}runs compare${OPT} <run_a_dir> <run_b_dir>${NC}"
    echo -e "  ${CMD}runs best${OPT} [dir] [--metric val_accuracy|val_loss|test_accuracy]${NC}"
    echo -e ""
    echo -e "${BOLD}Test Commands:${NC}"
    echo -e "  ${CMD}test${OPT} [opts]${NC}                 run test suite;"
    echo -e "    ${OPT}-m <mark>${NC}                 run tests matching a mark (e.g. -m 'not tf');"
    echo -e "    ${OPT}-c${NC}                        generate code coverage summary;"
    echo -e ""
    echo -e "${BOLD}Release Commands:${NC}"
    echo -e "  ${CMD}release${OPT} [--dry-run]${NC}          preview next version bump (CI handles the actual bump);"
}

action_init(){
    if [ -d .venv ]; then
        rm -rf .venv
    fi

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip -q
    pip install -e ".[dev]"
}

action_activate(){
    if [ ! -f .venv/bin/activate ]; then
        echo "Virtual environment not found. Run: ./helper.sh init"
        exit 1
    fi
    source .venv/bin/activate
}

action_generate(){
    action_activate
    generate "$@"
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

action_test(){
    action_activate
    OPTS=()
    while getopts ":m:c" opt; do
        case $opt in
            m) OPTS+=(-m "$OPTARG") ;;
            c) OPTS+=(--cov=core --cov=cli --cov=augmentations --cov-report=term-missing) ;;
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
    generate)
        action_generate ${@:2}
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
    test)
        action_test ${@:2}
        ;;
    release)
        action_release ${@:2}
        ;;
    *)
        action_usage
        ;;
esac

exit 0
