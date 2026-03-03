# Testing and linting

The npm wrappers already set `PYTHONPATH=.` so the AI Scientist package is
findable without extra shell juggling. Run the standard targets from the root
via `npm run lint`, `npm run format`, `npm run type`, and `npm run test` to
reach the same coverage as `PYTHONPATH=. npm run test` mentioned elsewhere in
this repo.
