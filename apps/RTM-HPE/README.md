# RTM-HPE

## Usage
```shell
./init.sh  # create .env
source .venv/bin/activate
gunicorn -c gunicorn_conf.py main:app
```

## Settings
- `config.yaml`
- `gunicorn.conf.py`

