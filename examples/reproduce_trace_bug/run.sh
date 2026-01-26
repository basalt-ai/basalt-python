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

# Change to the service directory
cd "$(dirname "$0")"

# Check if hatch environment exists
if ! hatch env show default &>/dev/null; then
    echo -e "${YELLOW}Creating hatch environment...${NC}"
    hatch env create
fi

# Install basalt-py from parent directory
echo -e "${YELLOW}Installing basalt-py from parent directory...${NC}"
hatch run pip install -q -e ../..

echo -e "${GREEN}Starting services...${NC}"
echo ""

# Start services in background
echo -e "${GREEN}Starting Service B on port 8001...${NC}"
hatch run service-b &
SERVICE_B_PID=$!

# Wait a moment for Service B to start
sleep 2

echo -e "${GREEN}Starting Service A on port 8000...${NC}"
hatch run service-a &
SERVICE_A_PID=$!

# Wait for services to fully start
sleep 2

# Trap to cleanup on exit
trap "echo -e '\n${YELLOW}Shutting down services...${NC}'; kill $SERVICE_A_PID $SERVICE_B_PID 2>/dev/null; exit 0" SIGINT SIGTERM EXIT

echo ""
echo -e "${GREEN}✓ Services started successfully!${NC}"
echo ""
echo "Service A: http://localhost:8000"
echo "Service B: http://localhost:8001"
echo ""
echo -e "${YELLOW}Test the repro:${NC}"
echo "  curl -X POST http://localhost:8000/call-service-b"
echo ""
echo -e "${RED}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait for processes
wait
