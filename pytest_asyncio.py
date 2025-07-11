import asyncio
import inspect

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark async test")


LOOP = asyncio.new_event_loop()

@pytest.hookimpl(tryfirst=True)


def pytest_pyfunc_call(pyfuncitem):
    testfunc = pyfuncitem.obj
    if inspect.iscoroutinefunction(testfunc):
        asyncio.set_event_loop(LOOP)
        kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
        LOOP.run_until_complete(testfunc(**kwargs))
        asyncio.set_event_loop(None)
        return True


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    func = fixturedef.func
    if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
        asyncio.set_event_loop(LOOP)
        params = {name: request.getfixturevalue(name) for name in fixturedef.argnames}
        if inspect.isasyncgenfunction(func):
            agen = func(**params)
            value = LOOP.run_until_complete(agen.__anext__())

            def finalizer() -> None:
                try:
                    LOOP.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    pass

            request.addfinalizer(finalizer)
            fixturedef.cached_result = (value, 0, None)
            return value
        result = LOOP.run_until_complete(func(**params))
        fixturedef.cached_result = (result, 0, None)
        return result
      
