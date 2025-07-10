import asyncio
import inspect
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark async test")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    testfunc = pyfuncitem.obj
    if inspect.iscoroutinefunction(testfunc):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
        loop.run_until_complete(testfunc(**kwargs))
        loop.close()
        asyncio.set_event_loop(None)
        return True


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    func = fixturedef.func
    if inspect.iscoroutinefunction(func):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        params = {name: request.getfixturevalue(name) for name in fixturedef.argnames}
        if inspect.isasyncgenfunction(func):
            agen = func(**params)
            value = loop.run_until_complete(agen.__anext__())

            def finalizer() -> None:
                try:
                    loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    pass
                loop.close()
                asyncio.set_event_loop(None)

            request.addfinalizer(finalizer)
            fixturedef.cached_result = (value, 0, None)
            return value
        result = loop.run_until_complete(func(**params))
        loop.close()
        asyncio.set_event_loop(None)
        fixturedef.cached_result = (result, 0, None)
        return result
      
