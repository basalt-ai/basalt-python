#!/usr/bin/env bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if direnv is loaded
if [ -z "$BASALT_API_KEY" ]; then
    echo -e "${RED}Error: Environment not loaded.${NC}"
    echo "Please run: ${YELLOW}direnv allow${NC}"
    echo "Then edit .envrc with your Basalt API key"
    exit 1
fi

echo -e "${GREEN}Environment loaded successfully${NC}"
echo "BASALT_ENVIRONMENT: $BASALT_ENVIRONMENT"
echo "BASALT_SERVICE_NAME: $BASALT_SERVICE_NAME"
echo ""

# Change to the script directory
cd "$(dirname "$0")"

# Check if hatch environment exists
if ! hatch env show default &>/dev/null; then
    echo -e "${YELLOW}Creating hatch environment...${NC}"
    hatch env create
fi

# Install basalt-py from parent directory
echo -e "${YELLOW}Installing basalt-py from parent directory...${NC}"
hatch run pip install -q -e ../..

echo -e "${GREEN}Running OpenAI example...${NC}"
echo ""

hatch run run
