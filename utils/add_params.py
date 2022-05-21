from argparse import ArgumentError


def safe_add_params(params, name, **kwargs):
    try:
        params.add_argument(name, **kwargs)
    except ArgumentError:
        pass
