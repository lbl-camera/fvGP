# Logging

The fvGP package uses the [Loguru](https://github.com/Delgan/loguru) library for sophisticated log management. This follows similar principles as the
vanilla Python logging framework, with additional functionality and performance benefits. You may want to enable logging
in interactive use, or for debugging purposes.

## Configuring logging

To enable logging in fvGP:

```python
from loguru import logger
logger.enable("fvgp")
```

To configure the logging level:

```python
logger.add(sys.stdout, filter="fvgp", level="INFO")
```
See [Python's reference on levels](https://docs.python.org/3/howto/logging.html) for more info.

To log to a file:

```python
logger.add("file_{time}.log")
```

Loguru provides many [further options for configuration](https://github.com/Delgan/loguru).