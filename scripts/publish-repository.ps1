$ErrorActionPreference = "Stop"

Write-Host "PART 0 // preflight"

if (-not (Test-Path "pyproject.toml")) {
    throw "Run this script from the atlas-model-lab repository root."
}

Get-Command git | Out-Null
Get-Command gh | Out-Null
Get-Command py | Out-Null
gh auth status

gh repo view AtlasReaper311/atlas-model-lab *> $null
if ($LASTEXITCODE -eq 0) {
    throw "AtlasReaper311/atlas-model-lab already exists. Refusing to recreate it."
}

Write-Host "PART 1 // validate"

py -3.12 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -e ".[dev]"
& ".\.venv\Scripts\python.exe" scripts\validate.py

Write-Host "PART 2 // initialise git"

git init -b main
git add .
git diff --cached --check
git status --short
git commit -m "Initial Atlas Model Lab foundation"

Write-Host "PART 3 // create GitHub repository"

gh repo create AtlasReaper311/atlas-model-lab --public --source=. --remote=origin --push --description "Experimental neural-network and transformer implementations built from first principles, with reproducible tests and benchmarks." --homepage "https://atlas-systems.uk"

Write-Host "PART 4 // apply repository metadata"

gh repo edit AtlasReaper311/atlas-model-lab --add-topic atlas-systems
gh repo edit AtlasReaper311/atlas-model-lab --add-topic python
gh repo edit AtlasReaper311/atlas-model-lab --add-topic local-ai
gh repo edit AtlasReaper311/atlas-model-lab --add-topic llm

Write-Host "PART 5 // apply Atlas status labels"

gh label create "status:blocked" --repo AtlasReaper311/atlas-model-lab --color d73a4a --description "Progress is blocked by a concrete dependency or failure."
gh label create "status:live-verified" --repo AtlasReaper311/atlas-model-lab --color 4ade80 --description "Live state has been independently or owner-verified after rollout."
gh label create "status:owner-review" --repo AtlasReaper311/atlas-model-lab --color f5a623 --description "Implementation is ready for Atlas owner review."
gh label create "status:rollout-pending" --repo AtlasReaper311/atlas-model-lab --color b7791f --description "Source is merged or approved; live rollout is still pending."
gh label create "status:superseded" --repo AtlasReaper311/atlas-model-lab --color 555560 --description "Work was replaced by a newer change and should not be merged."

Write-Host "PART 6 // verify repository"

gh repo view AtlasReaper311/atlas-model-lab --json nameWithOwner,visibility,defaultBranchRef,description,homepageUrl,repositoryTopics,url
gh label list --repo AtlasReaper311/atlas-model-lab --limit 100

Write-Host "Repository bootstrap complete."
Write-Host "Atlas Infra classification and provider-guard rollout remain separate follow-up gates."
