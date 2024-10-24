if [[ "${{ github.event.pull_request.base.ref }}" == "main"]]; then
    if [[ "${{ github.head_ref }}" != "dev" ]]; then
        echo "Pull requests to main branch must come from the dev branch."
        exit 1
    fi
fi
if [[ "${{ github.event.pull_request.base.ref }}" == "dev"]]; then
    exit 0
fi