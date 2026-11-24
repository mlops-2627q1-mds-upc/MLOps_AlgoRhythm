Build and execute one file:
```bash
docker compose run --rm --build algorythm-test python -m pytest tests/test_<test_name>.py -v
```

Execute all tests:
```bash
docker compose run --rm algorythm-test
```

Execute one test:
```bash
docker compose run --rm algorythm-test python -m pytest tests/test_<test_name>.py -v
```