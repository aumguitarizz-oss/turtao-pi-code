"""Shared lock serializing all calls into dlib via face_recognition.

dlib's HOG face detector is not safe to call concurrently from multiple
threads: two threads calling face_recognition.face_locations /
face_encodings at the same time corrupts its internal state and crashes
the process natively (SIGSEGV on macOS, "munmap_chunk(): invalid
pointer" / abort on Linux glibc). This app runs recognition continuously
on a background thread while enrollment can be triggered from a GUI or
API thread at any time, so every call into face_recognition must hold
this lock.
"""

import threading

DLIB_LOCK = threading.Lock()
