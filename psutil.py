class Process:
    def __init__(self, pid=None):
        pass

    class MemoryInfo:
        def __init__(self, rss=0):
            self.rss = rss

    def memory_info(self):
        return self.MemoryInfo(0)
