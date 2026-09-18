---
title: Logger
description: API reference for Logger.
---

# `Logger`

**Module:** `opytimizer.utils.logging`

A customized Logger file that enables the possibility of only logging to file.

## Constructor

```python
Logger(name, level=0)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `name` |  |  | — |
| `level` |  | `0` | — |

## Methods

### `addHandler`

```python
addHandler(self, hdlr)
```

Add the specified handler to this logger.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `hdlr` |  |  | — |

### `callHandlers`

```python
callHandlers(self, record)
```

Pass a record to all relevant handlers.

Loop through all handlers for this logger and its parents in the
logger hierarchy. If no handler was found, output a one-off error
message to sys.stderr. Stop searching up the hierarchy whenever a
logger with the "propagate" attribute set to zero is found - that
will be the last logger whose handlers are called.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `record` |  |  | — |

### `critical`

```python
critical(self, msg, *args, **kwargs)
```

Log 'msg % args' with severity 'CRITICAL'.

To pass exception information, use the keyword argument exc_info with
a true value, e.g.

logger.critical("Houston, we have a %s", "major disaster", exc_info=True)

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `msg` |  |  | — |
| `args` |  |  | — |
| `kwargs` |  |  | — |

### `debug`

```python
debug(self, msg, *args, **kwargs)
```

Log 'msg % args' with severity 'DEBUG'.

To pass exception information, use the keyword argument exc_info with
a true value, e.g.

logger.debug("Houston, we have a %s", "thorny problem", exc_info=True)

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `msg` |  |  | — |
| `args` |  |  | — |
| `kwargs` |  |  | — |

### `error`

```python
error(self, msg, *args, **kwargs)
```

Log 'msg % args' with severity 'ERROR'.

To pass exception information, use the keyword argument exc_info with
a true value, e.g.

logger.error("Houston, we have a %s", "major problem", exc_info=True)

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `msg` |  |  | — |
| `args` |  |  | — |
| `kwargs` |  |  | — |

### `exception`

```python
exception(self, msg, *args, exc_info=True, **kwargs)
```

Convenience method for logging an ERROR with exception information.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `msg` |  |  | — |
| `args` |  |  | — |
| `exc_info` |  | `True` | — |
| `kwargs` |  |  | — |

### `fatal`

```python
fatal(self, msg, *args, **kwargs)
```

Don't use this method, use critical() instead.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `msg` |  |  | — |
| `args` |  |  | — |
| `kwargs` |  |  | — |

### `findCaller`

```python
findCaller(self, stack_info=False, stacklevel=1)
```

Find the stack frame of the caller so that we can note the source
file name, line number and function name.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `stack_info` |  | `False` | — |
| `stacklevel` |  | `1` | — |

### `getChild`

```python
getChild(self, suffix)
```

Get a logger which is a descendant to this one.

This is a convenience method, such that

logging.getLogger('abc').getChild('def.ghi')

is the same as

logging.getLogger('abc.def.ghi')

It's useful, for example, when the parent logger is named using
__name__ rather than a literal string.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `suffix` |  |  | — |

### `getChildren`

```python
getChildren(self)
```

### `getEffectiveLevel`

```python
getEffectiveLevel(self)
```

Get the effective level for this logger.

Loop through this logger and its parents in the logger hierarchy,
looking for a non-zero logging level. Return the first one found.

### `handle`

```python
handle(self, record)
```

Call the handlers for the specified record.

This method is used for unpickled records received from a socket, as
well as those created locally. Logger-level filtering is applied.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `record` |  |  | — |

### `hasHandlers`

```python
hasHandlers(self)
```

See if this logger has any handlers configured.

Loop through all handlers for this logger and its parents in the
logger hierarchy. Return True if a handler was found, else False.
Stop searching up the hierarchy whenever a logger with the "propagate"
attribute set to zero is found - that will be the last logger which
is checked for the existence of handlers.

### `info`

```python
info(self, msg, *args, **kwargs)
```

Log 'msg % args' with severity 'INFO'.

To pass exception information, use the keyword argument exc_info with
a true value, e.g.

logger.info("Houston, we have a %s", "notable problem", exc_info=True)

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `msg` |  |  | — |
| `args` |  |  | — |
| `kwargs` |  |  | — |

### `isEnabledFor`

```python
isEnabledFor(self, level)
```

Is this logger enabled for level 'level'?

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `level` |  |  | — |

### `log`

```python
log(self, level, msg, *args, **kwargs)
```

Log 'msg % args' with the integer severity 'level'.

To pass exception information, use the keyword argument exc_info with
a true value, e.g.

logger.log(level, "We have a %s", "mysterious problem", exc_info=True)

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `level` |  |  | — |
| `msg` |  |  | — |
| `args` |  |  | — |
| `kwargs` |  |  | — |

### `makeRecord`

```python
makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None)
```

A factory method which can be overridden in subclasses to create
specialized LogRecords.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `name` |  |  | — |
| `level` |  |  | — |
| `fn` |  |  | — |
| `lno` |  |  | — |
| `msg` |  |  | — |
| `args` |  |  | — |
| `exc_info` |  |  | — |
| `func` |  | `None` | — |
| `extra` |  | `None` | — |
| `sinfo` |  | `None` | — |

### `removeHandler`

```python
removeHandler(self, hdlr)
```

Remove the specified handler from this logger.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `hdlr` |  |  | — |

### `setLevel`

```python
setLevel(self, level)
```

Set the logging level of this logger.  level must be an int or a str.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `level` |  |  | — |

### `to_file`

```python
to_file(self, msg: str, *args, **kwargs) -> None
```

Logs the message only to the logging file.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `msg` | `str` |  | Message to be logged. |
| `args` |  |  | — |
| `kwargs` |  |  | — |

### `warn`

```python
warn(self, msg, *args, **kwargs)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `msg` |  |  | — |
| `args` |  |  | — |
| `kwargs` |  |  | — |

### `warning`

```python
warning(self, msg, *args, **kwargs)
```

Log 'msg % args' with severity 'WARNING'.

To pass exception information, use the keyword argument exc_info with
a true value, e.g.

logger.warning("Houston, we have a %s", "bit of a problem", exc_info=True)

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `msg` |  |  | — |
| `args` |  |  | — |
| `kwargs` |  |  | — |
