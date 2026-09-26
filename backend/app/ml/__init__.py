"""Optional deep-learning backends. Each module degrades gracefully: if its
dependencies or weights are missing, `available()` returns False and the
classical tool is used instead."""
