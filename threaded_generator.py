"""https://github.com/Lasagne/Lasagne/issues/12#issuecomment-59494251"""


def threaded_generator(generator, num_cached=50):
    from queue import Queue

    qu = Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            qu.put(item)
        qu.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = qu.get()
    while item is not sentinel:
        yield item
        qu.task_done()
        item = qu.get()
