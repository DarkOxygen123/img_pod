# Testing

This folder generates two local test selfies and calls the CPU interface `POST /v1/profile/create` to save:
- `*_features.json`
- `*_profile.png`

## Setup

1) Copy config:
- `cp testing/config.example.json testing/config.json`

2) Set `interface_base_url` in `testing/config.json`, e.g.:
- `https://f2cscss4pwa7ig-8000.proxy.runpod.net`

## Run

- Generate selfies:
  - `.venv/bin/python testing/generate_test_selfies.py`

- Call profile endpoint and save outputs:
  - `.venv/bin/python testing/profile_create_and_save.py`

Outputs go to `testing/output/`.
