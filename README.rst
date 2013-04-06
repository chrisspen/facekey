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

You should now have several images in your ~/.facekey/pending folder.
Manually review each one, and rename it with a unique identifier corresponding to the person's name. (e.g. an image of "Bob Jones" should be renamed something like "bobjones.png")
Delete images that:

* Do not contain a person's entire face.
* Are of poor quality (blurry, grainy, dark, low contrast, etc). Shadows or slight angles are fine.

Denote images of the same person with an incrementing index (e.g. bobjones1.png, bobjones2.png, bobjones3.png, etc).
Then copy these images into your ~/.facekey/gallery.

Step 3: Train classifier.

Now that we have a large collection of tagged images, we need to train our Eigenfaces classifier so it can identify a person given a new image. To do this, run:

    facekey.py train

Step 4: Test classifier. *incomplete*

Assuming everything has worked correctly, the classifier should now be able to recognize your face. Ensure you have a webcam attached and run:

    facekey.py classify --realtime

You should now see a small GUI window showing your webcam video feed. If your face is detected in the feed, it should be outlined with a colored box labeled with your name.

Step 5: Classify images.

You can also use this script from the command line to detect and identify faces in untagged images by running:

    facekey.py classify <image directory>
    
Or classify one of more specific images with:

    facekey.py classify <image1> <image2> ... <imageN>

Detected faces will be displayed to standard output in the following CSV format:

    filename,x,y,width,height,name,dist

Step 6: Unlock desktop. *incomplete*

Specify the name you used to identify your own images in ~/.facekey/unlockers.

Then run:

    facekey.py run
    
Lock your desktop, then position your face infront of your webcam and wait. After a few seconds, your desktop should become unlocked.

If it is not unlocked, that means the webcam wasn't able to detect a face in the video stream, or was able to detect a face, by mis-classified it as someone else.

Check your ~/.facekey/probes folder. Captured face images will be placed in this folder. Locate images of you and process them according to the tagging and training steps.

Once you're comfortable with the performance and accuracy of the script, you can install it to run automatically via:

    facekey.py install
    
Caveats
-------

1. Detecting the unknown

When classifying faces, it can be difficult to detect an unknown face.
The algorithm works by finding the known face that is the most similar
to the detected face. The metric of similarity is known as the "distance".
A distance of 0 indicates a perfect match to a known face.
A large distance indicates a less known face.
We classify a face as "unknown" if this distance is above some threshold.
However, what threshold is appropriate for your images depends largely on
what images you've already trained and the images being classified.

One way to determine this threshold is to download faces of random people
that definitely should not be recognized, run the classify command,
and then use the average or minimum distance as the known/unknown threshold.

2. Slow performance

The bulk of the Eigenfaces algorithm uses Numpy to improve speed.
Unfortunately, detection and classification of a face can still be painfully
slow. Testing on my personal machine took on average 7 seconds to recognize my
face and unlock my desktop. Depending on your needs and your system's speed,
this may be unusably slow.

At some point I'd like to replace the Python/Numpy Eigenfaces implementation
with the pure `C++ implementation in OpenCV
http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html#eigenfaces`_.
However, as of this authoring, that implementation, and its Python bindings,
are still incomplete. 