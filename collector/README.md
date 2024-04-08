# Port Forwarding Services Are Forwarding Security Risks

## PFW Collector

### Dependencies

**Note**: The public `time_machine_pipeline.py` does NOT support passive DNS feature. If you want to run it correctly, you need to create a file named `get_subdomain.py`.

You can use `pip install -r requirements.txt` to install all the dependencies (no need for `requests` if just using `time_machine.py` independently). You may need to install Firefox for Playwright, by `playwright install firefox`.

### Code Details

1. `time_machine.py`: core program for snapshotting the website(s), with the screenshots and the network traces.
   + You can use `python3 time_machine.py "ngrok.com" "ngrok" "./test/" --is_domain` for test. It will create a new directory named `test` (if it doesn't already exist) to store the collected snapshots.
2. `time_machine_pipeline.py`: the pipeline containing PFW domain names discovering (not publicly accessed), aliveness test, and PFW snapshotting.

### Usage

```bash
# get_subdomain.py is needed by time_machine_pipeline.py, details are in the section Dependencies.
touch get_subdomain.py
python3 time_machine_pipeline.py "./config_template.yaml" --date_dir "./test/" --alive_seed_file <(echo "ngrok.com") --steps 6
```

And you should expect to see results in a few seconds:

```
test
├── alive_test
│  └── 20240101
│     └── alive_test_results.json
├── screenshots
│  └── 20240101
│     ├── ngrok
│     │  ├── http_ngrok.com
│     │  │  ├── page_screenshot.png
│     │  │  └── test.har
│     │  └── https_ngrok.com
│     │     ├── page_screenshot.png
│     │     └── test.har
│     └── result_stats.json
└── subdomains
   └── 20240101
```

**Notes**:
+ You can get help by using `-h` option, e.g., `python3 time_machine.py -h`.
+ `config_template.yaml` should be updated, especially `provider` field.
+ `--alive_seed_file` should be provided by a file with domains you want to snapshot.
+ Option `--data_dir` is prioritized over the `data_dir` field in `config_template.yaml`.
+ You can select the step(s) you want `time_machine_pipeline.py` to do by option `--steps`.
