import asyncio
import sqlite3


class Row(sqlite3.Row):
    pass

async def connect(dsn, uri=False, timeout=30):
    conn = sqlite3.connect(dsn, timeout=timeout, uri=uri, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return Connection(conn)

class Connection:
    def __init__(self, conn):
        self._conn = conn
        self.row_factory = conn.row_factory

    async def execute(self, sql, params=()):
        return Cursor(self._conn.execute(sql, params))

    async def commit(self):
        self._conn.commit()

    async def close(self):
        self._conn.close()

    # The Connection object in this stub does not expose fetch methods
    # directly; callers should use the Cursor returned by ``execute``.

class Cursor:
    def __init__(self, cur):
        self._cur = cur

    async def fetchone(self):
        return self._cur.fetchone()

    async def fetchall(self):
        return self._cur.fetchall()
