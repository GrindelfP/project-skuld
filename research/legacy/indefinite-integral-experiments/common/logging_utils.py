import io

_log_buffer = io.StringIO()


def tee_print(*args, **kwargs):
    """Print to console and accumulate in _log_buffer."""
    print(*args, **kwargs)
    # Replicate to buffer
    kwargs_buf = {k: v for k, v in kwargs.items() if k != 'file'}
    print(*args, **kwargs_buf, file=_log_buffer)
