=============================================================================
Facekey - Unlock your desktop with your face.
=============================================================================

Overview
--------

**The functionality is currently incomplete.**

This script runs as a daemon (background process) and when it detects that
you've locked your deskop, it starts monitoring your webcam. If it detects your
face in the webcam image, it automatically unlocks your desktop without
requiring you to enter your password.

It can also function as a general-purpose command line tool for collecting
and identifying faces found in image files.

This script relies heavily on `OpenCV
<www.opencv.org>`_ for face detection and a modified version
of `Pyfaces
<http://code.google.com/p/pyfaces/>`_ for facial recognition.

Installation
------------

Install the Python Image Library:

    sudo apt-get install python-imaging

Install OpenCV:

    sudo apt-get install python-opencv libcv2.3

Install Python bindings for dbus and gobject:

    sudo apt-get install python-dbus python-gobject

Install script package:

    sudo pip install https://github.com/chrisspen/facekey/zipball/master

Usage
-----

Step 1: Collect images.

In order for the script to recognize your face, it needs:

* examples of what you look like
* examples of what you don't look like

Start by finding a folder on your computer containing several high-quality images of you and other people. Then run:

    facekey.py collect <image directory>

This will recursively search for all image files, attempt to detect faces in each image, and save each valid face to a separate image file.
A few notes:

* The face detector uses the pre-trained frontal-face Haar cascade filter in OpenCV. It's fairly fast, but if your images are several MBs in size, it can take 10-15 seconds to process each image.
* Accuracy of the face detector is good, but not perfect. It will occassionally miss some faces or identify non-face regions as faces. But that's ok, because you'll filter out those mistakes in the next step.
* Not all correctly detected faces will be useful. If a face is too small then it will be ignored and no file will be saved for it because it won't have enough information to contribute to the face classifier.

Step 2: Tag images.

You should now have several images in your <base_image_dir>/pending folder.
Manually review each one, and rename it with a unique identifier corresponding to the person's name. (e.g. an image of "Bob Jones" should be renamed something like "bobjones.png")
Delete images that:
* Do not contain a person's entire face.
* Are of poor quality (blurry, grainy, dark, low contrast, etc). Shadows or slight angles are fine.
Denote images of the same person with an incrementing index (e.g. bobjones1.png, bobjones2.png, bobjones3.png, etc).
Then copy these images into your <base_image_dir>/gallery.

Step 3: Train classifier.

Now that we have a large collection of tagged images, we need to train our Eigenfaces classifier so it can identify a person given a new image. To do this, run:

    facekey.py train

Step 4: Test classifier.

Assuming everything has worked correctly, the classifier should now be able to recognize your face. Ensure you have a webcam attached and run:

    facekey.py classify --realtime

You should now see a small GUI window showing your webcam video feed. If your face is detected in the feed, it should be outlined with a colored box labeled with your name.

Step 5: Tag images.

You can also use this script from the command line to detect and identify faces in untagged images by running:

    facekey.py classify <image directory>

Detected faces will be displayed to standard output in the following CSV format:

    filename,x,y,width,height,name
    