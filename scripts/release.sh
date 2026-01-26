#!/bin/bash
set -e

# Script pour automatiser le processus de release
# Usage: ./scripts/release.sh [patch|minor|major|VERSION]

if [ $# -eq 0 ]; then
    echo "Usage: $0 [patch|minor|major|VERSION]"
    echo ""
    echo "Examples:"
    echo "  $0 patch      # 1.1.1 -> 1.1.2"
    echo "  $0 minor      # 1.1.1 -> 1.2.0"
    echo "  $0 major      # 1.1.1 -> 2.0.0"
    echo "  $0 1.2.3      # Set specific version"
    exit 1
fi

BUMP_TYPE=$1

# Couleurs pour l'output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting release process...${NC}\n"

# Vérifier qu'on est sur master et que le working directory est propre
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "master" ]; then
    echo -e "${RED}❌ Error: You must be on the master branch${NC}"
    echo "Current branch: $BRANCH"
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    echo -e "${RED}❌ Error: Working directory is not clean${NC}"
    echo "Please commit or stash your changes first"
    git status --short
    exit 1
fi

# Pull latest changes
echo -e "${YELLOW}📥 Pulling latest changes from origin...${NC}"
git pull origin master

# Run tests
echo -e "${YELLOW}🧪 Running tests...${NC}"
if ! hatch run test; then
    echo -e "${RED}❌ Tests failed! Aborting release.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Tests passed${NC}\n"

# Get current version
CURRENT_VERSION=$(hatch version)
echo -e "${BLUE}Current version: ${CURRENT_VERSION}${NC}"

# Bump version
echo -e "${YELLOW}📝 Bumping version...${NC}"
hatch version $BUMP_TYPE
NEW_VERSION=$(hatch version)
echo -e "${GREEN}New version: ${NEW_VERSION}${NC}\n"

# Confirm with user
echo -e "${YELLOW}This will:${NC}"
echo "  1. Commit the version change"
echo "  2. Create and push tag v${NEW_VERSION}"
echo "  3. Trigger GitHub Actions to publish to PyPI"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}❌ Release cancelled${NC}"
    # Revert version change
    git checkout basalt/_version.py
    exit 1
fi

# Commit version change
echo -e "${YELLOW}💾 Committing version change...${NC}"
git add basalt/_version.py
git commit -m "chore: bump version to ${NEW_VERSION}"

# Push commit
echo -e "${YELLOW}📤 Pushing commit to origin...${NC}"
git push origin master

# Create and push tag
echo -e "${YELLOW}🏷️  Creating and pushing tag v${NEW_VERSION}...${NC}"
git tag -a "v${NEW_VERSION}" -m "Release version ${NEW_VERSION}"
git push origin "v${NEW_VERSION}"

echo ""
echo -e "${GREEN}✨ Release process completed!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Monitor GitHub Actions: https://github.com/basalt-ai/basalt-python/actions"
echo "  2. Check PyPI: https://pypi.org/project/basalt-sdk/${NEW_VERSION}/"
echo "  3. Review GitHub Release: https://github.com/basalt-ai/basalt-python/releases/tag/v${NEW_VERSION}"
echo ""
echo -e "${YELLOW}⏳ The package should be available on PyPI in a few minutes${NC}"
