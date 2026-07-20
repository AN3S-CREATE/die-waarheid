# Continuous Integration

The `CI` workflow validates Python quality and tests, frontend lint/build, application security, and the production Dockerfile. It runs on pushes and pull requests for `main` and `develop`, every Monday at 03:17 UTC, and on manual dispatch. Pull requests use BuildKit's fast Dockerfile validation call; trusted push, schedule, and manual runs execute the full cached image build.

`CI Health` is the stable aggregate check intended for branch protection. It fails unless every required job succeeds.

## Required versions

- Python 3.11 for the primary quality and security jobs.
- Python 3.10, 3.11, and 3.12 for the compatibility test matrix.
- Node.js 24 with the committed `frontend/package-lock.json`.
- Docker with BuildKit for the container build.

CI-only Python tools are pinned in `requirements-ci.txt`. Application and test dependencies remain in `requirements.txt` and `test-requirements.txt`.

## Run CI locally

Create and activate a Python 3.11 virtual environment, then run:

```bash
python -m pip install -r requirements-ci.txt
python -m compileall -q die_waarheid
flake8 die_waarheid --count --select=E9,F63,F7,F82 --show-source --statistics
black --check die_waarheid
isort --check-only die_waarheid
mypy die_waarheid/config.py die_waarheid/src/models.py die_waarheid/src/chat_parser.py die_waarheid/src/evidence_scoring.py die_waarheid/src/risk_escalation_matrix.py --ignore-missing-imports --disable-error-code=import-untyped --no-incremental
bandit -r die_waarheid -s B101,B102,B301 -ll -ii -f json -o bandit-report.json
```

The mypy list is an explicit incremental typed-core boundary. New modules should be added as their existing type debt is resolved; do not replace this gate with a global error suppression.

Install the application tests and run the same coverage gate:

```bash
python -m pip install -r requirements.txt
python -m pip install -r test-requirements.txt
pytest tests/ --cov=die_waarheid --cov-report=xml --cov-report=term-missing --junitxml=junit.xml
```

Validate the frontend and container:

```bash
cd frontend
npm ci --no-audit
npm run lint
npm run build
cd ..
docker build --tag die-waarheid:ci .
```

If [`act`](https://github.com/nektos/act) and Docker are installed, use a clean pull-request simulation:

```bash
act pull_request -W .github/workflows/ci-cd.yml
```

`act` is a useful approximation, but GitHub-hosted runner images, caches, and service behavior must still be verified in GitHub Actions.

## Secrets and variables

No secrets or repository variables are required for CI. The container job builds but never logs in or pushes, making push and fork pull-request runs deterministic and credential-safe.

Image publication and deployment are deliberately not represented by echo-only placeholder jobs. When a real release workflow is added, use a protected GitHub Environment and document its registry/deployment credentials here. Never expose environment secrets to fork pull requests.

The application itself may require runtime values such as `API_KEY`; those values are not needed by the current unit tests and must not be stored in workflow YAML or uploaded artifacts.

## Reports and retention

- Python coverage XML and JUnit reports: 14 days.
- Frontend production bundle: 14 days.
- Bandit and dependency-audit JSON reports: 30 days.

Bandit blocks medium/high-severity findings with medium/high confidence. `pip-audit` is currently informational because the legacy exact dependency set needs a reviewed remediation baseline; findings create a warning annotation and remain available in the uploaded report.

## Re-run and debug a failure

From the Actions page, open the failed `CI` run and select **Re-run failed jobs**. From GitHub CLI:

```bash
gh run list --workflow CI --limit 10
gh run view RUN_ID --log-failed
gh run rerun RUN_ID --failed
```

For a pull request:

```bash
gh pr checks PR_NUMBER
```

Common failures:

- **Black or isort:** run `isort die_waarheid` followed by `black die_waarheid`, then commit the result.
- **Mypy:** fix the reported typed-core error. Do not remove the module or weaken the whole gate.
- **Bandit:** inspect `bandit-report.json`; fix genuine findings or add a narrowly scoped `# nosec TEST_ID` only with a code comment explaining why the behavior is safe.
- **Pytest:** download the matching `python-test-reports-*` artifact and inspect its JUnit XML and coverage output.
- **Frontend:** reproduce with `npm ci --no-audit && npm run lint && npm run build` from `frontend/`.
- **Container:** run the local Docker build with `--progress=plain`. Check dependency download, disk capacity, and `.dockerignore` before changing runner size.
- **Cache:** caches are performance optimizations only. Re-run once or delete the relevant Actions cache if corruption is suspected.

## Branch protection

Configure `main` to require the `CI Health` status check, require the branch to be up to date before merging, and require pull-request review. Do this only after the new workflow has completed once so GitHub knows the check name.
