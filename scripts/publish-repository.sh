#!/usr/bin/env bash
set -eu

# PART 0: preflight
echo "PART 0 // preflight"

if [ ! -f "pyproject.toml" ]; then
  echo "Run this script from the atlas-model-lab repository root." >&2
  exit 1
fi

command -v git >/dev/null
command -v gh >/dev/null
command -v python3 >/dev/null
gh auth status

if gh repo view AtlasReaper311/atlas-model-lab >/dev/null 2>&1; then
  echo "AtlasReaper311/atlas-model-lab already exists. Refusing to recreate it." >&2
  exit 1
fi

# PART 1: local environment and validation
echo "PART 1 // validate"

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python scripts/validate.py

# PART 2: local Git bootstrap
echo "PART 2 // initialise git"

git init -b main
git add .
git diff --cached --check
git status --short
git commit -m "Initial Atlas Model Lab foundation"

# PART 3: create the GitHub repository and push the reviewed bootstrap
echo "PART 3 // create GitHub repository"

gh repo create AtlasReaper311/atlas-model-lab --public --source=. --remote=origin --push --description "Experimental neural-network and transformer implementations built from first principles, with reproducible tests and benchmarks." --homepage "https://atlas-systems.uk"

# PART 4: required public metadata
echo "PART 4 // apply repository metadata"

gh repo edit AtlasReaper311/atlas-model-lab --add-topic atlas-systems
gh repo edit AtlasReaper311/atlas-model-lab --add-topic python
gh repo edit AtlasReaper311/atlas-model-lab --add-topic local-ai
gh repo edit AtlasReaper311/atlas-model-lab --add-topic llm

# PART 5: Atlas status labels
echo "PART 5 // apply Atlas status labels"

gh label create "status:blocked" --repo AtlasReaper311/atlas-model-lab --color d73a4a --description "Progress is blocked by a concrete dependency or failure."
gh label create "status:live-verified" --repo AtlasReaper311/atlas-model-lab --color 4ade80 --description "Live state has been independently or owner-verified after rollout."
gh label create "status:owner-review" --repo AtlasReaper311/atlas-model-lab --color f5a623 --description "Implementation is ready for Atlas owner review."
gh label create "status:rollout-pending" --repo AtlasReaper311/atlas-model-lab --color b7791f --description "Source is merged or approved; live rollout is still pending."
gh label create "status:superseded" --repo AtlasReaper311/atlas-model-lab --color 555560 --description "Work was replaced by a newer change and should not be merged."

# PART 6: provider read-back
echo "PART 6 // verify repository"

gh repo view AtlasReaper311/atlas-model-lab --json nameWithOwner,visibility,defaultBranchRef,description,homepageUrl,repositoryTopics,url
gh label list --repo AtlasReaper311/atlas-model-lab --limit 100

echo "Repository bootstrap complete."
echo "Atlas Infra classification and provider-guard rollout remain separate follow-up gates."
