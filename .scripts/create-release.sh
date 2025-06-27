#!/bin/bash
# Script to create a GitHub release for SynApps

# Configuration
VERSION="v0.4.0-alpha"
RELEASE_TITLE="SynApps v0.4.0 Alpha"
RELEASE_BRANCH="master"
REPO_URL="https://github.com/nxtg-ai/SynApps-v0.4.0"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}SynApps Release Creation Script${NC}"
echo "============================="
echo -e "Creating release: ${GREEN}$RELEASE_TITLE${NC}"

# Ensure we're on the right branch
echo -e "\n${YELLOW}Checking current branch...${NC}"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "$RELEASE_BRANCH" ]; then
  echo -e "${RED}Error: You must be on the $RELEASE_BRANCH branch to create a release.${NC}"
  echo -e "Current branch: $CURRENT_BRANCH"
  echo -e "Please switch to $RELEASE_BRANCH and try again."
  exit 1
fi

# Make sure we have the latest changes
echo -e "\n${YELLOW}Pulling latest changes from remote...${NC}"
git pull origin $RELEASE_BRANCH

# Check if there are any uncommitted changes
echo -e "\n${YELLOW}Checking for uncommitted changes...${NC}"
if ! git diff-index --quiet HEAD --; then
  echo -e "${RED}Error: You have uncommitted changes.${NC}"
  echo "Please commit or stash your changes before creating a release."
  exit 1
fi

# Create and push the tag
echo -e "\n${YELLOW}Creating and pushing tag $VERSION...${NC}"
git tag -a $VERSION -m "$RELEASE_TITLE"
git push origin $VERSION

echo -e "\n${GREEN}Tag $VERSION has been created and pushed to the repository.${NC}"
echo -e "Now go to: ${YELLOW}$REPO_URL/releases/new${NC}"
echo -e "Select the tag: ${GREEN}$VERSION${NC}"
echo -e "Title: ${GREEN}$RELEASE_TITLE${NC}"
echo -e "Description: Copy the content from ${GREEN}RELEASE_NOTES.md${NC}"
echo -e "\n${GREEN}Release process initiated successfully!${NC}"
