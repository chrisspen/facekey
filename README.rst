=============================================================================
Facekey - A Linux daemon that unlocks your desktop with your face.
=============================================================================

Overview
--------

This script runs as a daemon (background process) and when it detects that
you've locked your deskop, it starts monitoring your webcam. If it detects your
face in the webcam image, it automatically unlocks your desktop without
requiring you to enter your password.

This script relies heavily on `OpenCV
<www.opencv.org>`_ for face detection and a modified version
of `Pyfaces
<http://code.google.com/p/pyfaces/>`_ for facial recognition.

Installation
------------

Install OpenCV.

On Ubuntu, you might try:

    sudo apt-get install python-opencv livcv*

However, this script is only tested with OpenCV 2.3.1, so an earlier version is
unlikely to work.

Install Python bindings for dbus and gobject. Again, you can do this on Ubuntu
via:

    sudo apt-get install python-dbus python-gobject

Install script package:

    pip install 

Usage
-----

In order for the script to recognize your face, it needs some samples images of you.

You can record this with the script by running:

    facekey.py --train
