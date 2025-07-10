def safe_load(stream):
    # return minimal logging config to satisfy dictConfig
    return {"version": 1, "handlers": {"null": {"class": "logging.NullHandler"}}, "root": {"handlers": ["null"]}}
