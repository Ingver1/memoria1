CONTENT_TYPE_LATEST = 'text/plain; version=0.0.4; charset=utf-8'

class _Metric:
    def __init__(self, *args, **kwargs):
        self._value = 0

    def inc(self, amount=1):
        self._value += amount

    def set(self, value):
        self._value = value

    def time(self):
        class _Timer:
            def __enter__(self_inner):
                return None
            def __exit__(self_inner, exc_type, exc, tb):
                pass
        return _Timer()

class Counter(_Metric):
    pass
class Gauge(_Metric):
    pass
class Histogram(_Metric):
    pass

def generate_latest():
    return b""
