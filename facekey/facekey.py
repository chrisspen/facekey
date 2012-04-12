#!/usr/bin/python
"""
2012.3.14 CKS
A simple script to monitor the webcam while the machine is locked, and
automatically unlock when the user's face is detected.
"""
VERSION = (0, 1, 0)
__version__ = '.'.join(map(str, VERSION))

import atexit
import datetime
import os
import re
import sys
import time
import threading
import traceback

import dbus
import dbus.decorators
import dbus.mainloop.glib
dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

import gobject
# Necessary to prevent Python threads from freezing due to the GIL
# when gobject blocks while making system calls.
gobject.threads_init()

try:
    import cv
except ImportError, e:
    print(("Unable to import OpenCV Python extensions. Ensure OpenCV is " +
        "installed and is compiled to support Python extensions.\n%s") % (e,))
    sys.exit(1)

from daemon import Daemon
import pyfaces

ACTIONS = (
    INSTALL,
    UNINSTALL,
    TRAIN,
    START,
    STOP,
    RESTART,
    RUN,
) = (
    'install',
    'uninstall',
    'train',
    'start',
    'stop',
    'restart',
    'run',
)
DEFAULT_ACTION = START
DEFAULT_PIDFILE = '/tmp/facekey.pid'

OPENCV_DIR = '/usr/local/share/OpenCV'
DEFAULT_CASCADE = "haarcascades/haarcascade_frontalface_alt.xml"
DEFAULT_IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'images')

class FaceKey(Daemon):
    
    def __init__(self,
        key_name='system-owner',
        gui=False,
        opencv_dir=OPENCV_DIR,
        cascade=DEFAULT_CASCADE,
        images_dir=DEFAULT_IMAGES_DIR,
        eigen_faces=6,
        eigen_threshold=3,
        *args, **kwargs):
        super(FaceKey, self).__init__(*args, **kwargs)
        
        self.loop = None
        self._cam_thread = None
        atexit.register(self._atexit)
        
        self.locked = False
        self.gui = gui
        self.images_dir = images_dir
        
        # Configure OpenCV.
        self.target_width = 125
        self.target_height = 150
        self.saved_images = 0
        self.base = OPENCV_DIR
        assert os.path.isdir(self.base), \
            "Base OpenCV directory %s does not exist." % (self.base,)
        self.cascade = os.path.join(self.base, cascade)
        assert os.path.isfile(self.cascade), \
            "Face cascade file %s does not exist." % (self.cascade,)
        self.face_cascade = cv.Load(self.cascade)
        self.key_name = key_name
        
        # Configure PyFaces.
        self.pyf = pyfaces.PyFaces(
            imgsdir=os.path.join(images_dir, 'gallery'),
            egfnum=eigen_faces,
            thrsh=eigen_threshold,
            extn='png')
        
        # Configure camera monitoring thread.
        self._monitoring_camera = True
        self._cam_thread = threading.Thread(target=self._monitor_cam, args=())
        self._cam_thread.setDaemon(True)
        
        # Configure dbus.
        bus = self.bus = dbus.SessionBus()
        try:
            screensaver = self.screensaver = bus.get_object('org.gnome.ScreenSaver','/org/gnome/ScreenSaver')
            screensaver.connect_to_signal('ActiveChanged', self._on_screensaver_change)
        except dbus.DBusException:
            traceback.print_exc()
            sys.exit(1)
    
    @property
    def has_training(self):
        for fn in os.listdir(os.path.join(self.images_dir, 'gallery')):
            if self.key_name in fn:
                return True
        return False
    
    def _atexit(self):
        if self._cam_thread:
            self._monitoring_camera = False
        if self.loop:
            self.loop.quit()

    def _iter_face_names(self, image):
        """
        Detects faces in the image and attempts to identify them.
        Returns a generator iterating over all names of identified faces.
        """
        image_size = cv.GetSize(image)
    
        # Convert to grayscale.
        grayscale = cv.CreateImage(image_size, 8, 1)
        cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)
        #cv.EqualizeHist(grayscale, grayscale)
        _image = image
        image = grayscale
        
        t0 = time.time()
        faces = cv.HaarDetectObjects(
            image, 
            self.face_cascade,
            cv.CreateMemStorage(0), 1.2, 2, 0, (20, 20))
        t1 = time.time() - t0
        print 'haar secs:',t1
     
        if faces:
            print 'Face detected!'
            for face in faces:
                
                rect,neighbors = face
                _x, _y, _width, _height = rect
                height = _height
                width = cv.Round(self.target_width/float(self.target_height)*height)
                y = _y
                x = _x + cv.Round(abs(width-_width)/2.)
#                print width, height
                
                if width >= self.target_width and height >= self.target_height:
                    # Crop.
                    cropped = cv.CreateImage((width, height), image.depth, image.nChannels)
                    src_region = cv.GetSubRect(image, (x, y, width, height))
                    cv.Copy(src_region, cropped)
                    
                    # Resize.
                    #thumbnail = cv.CreateMat(target_height, target_width, cv.CV_8UC3)#color
                    thumbnail = cv.CreateMat(self.target_height, self.target_width, cv.CV_8UC1)#color
                    cv.Resize(cropped, thumbnail)
                    
                    # Save.
                    #TODO:keep this in memory?
                    fqfn = os.path.join(self.images_dir, "probes/unknown%03i.png" % self.saved_images)
                    cv.SaveImage(fqfn, thumbnail)
                    self.saved_images += 1
                    
                    # Identify face.
                    t0 = time.time()
                    name = self.pyf.match_name(fqfn)
                    t1 = time.time() - t0
                    print 'pyfaces secs:',t1
                    if name:
                        print '-'*80
                        print 'RECOGNIZED:',name
                        yield name
                    else:
                        print '-'*80
                        print 'NO RECOGNITION'
                    #os.remove(fqfn)
                
                # Indicate area containing face.
                if self.gui:
                    cv.Rectangle(_image, (x,y), (x+width,y+height), cv.RGB(255,0,0))
    
    def _monitor_cam(self):
        
        # Create windows.
        if self.gui:
            cv.NamedWindow('Camera', cv.CV_WINDOW_AUTOSIZE)
     
        # Create capture device.
        device = 0 # assume we want first device
        capture = cv.CreateCameraCapture(0)
        w,h = 640,480
        #w,h = 320,240
        cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, w)
        cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, h)
     
        # Check if capture device is OK.
        if not capture:
            print "Error opening capture device"
            sys.exit(1)
        
        print "Monitoring camera..."
        while self._monitoring_camera:
            if not self.locked:
#                print 'Not locked. Sleeping.'
                time.sleep(1)
                continue
     
            # Capture an image from the camera.
            frame = cv.QueryFrame(capture)
            if frame is None:
                break
            cv.Flip(frame, None, 1)
     
            # Detect and recognize faces.
            faces_found = False
            for name in self._iter_face_names(frame):
                faces_found = True
                if name == self.key_name:
                    self.unlock()
            if faces_found:
                time.sleep(1)
     
            # Display webcam image.
            if self.gui:
                cv.ShowImage('Camera', frame)
         
                # handle events
                delay_ms=1
                k = cv.WaitKey(delay_ms)
         
                if k == 0x1b: # ESC
                    print 'ESC pressed. Exiting ...'
                    break
    
    def _on_screensaver_change(self, locked):
        self.locked = bool(locked)
        if self.locked:
            self.last_datetime_locked = datetime.datetime.now()
            #TODO:delay autounlock after N seconds
    
    def run(self):
        assert self.has_training, \
            ("No training images detected. Please run `%s train` to train " +
             "the script to recognize your face.") % (__file__)
        self._cam_thread.start()
        print "Monitoring screensaver..."
        loop = self.loop = gobject.MainLoop()
        loop.run()
    
    def unlock(self):
        if not self.locked:
            return
        self.screensaver.SetActive(False)

#    if action == INSTALL:
#        todo
#        # Copy symlink to /etc/init.d
#    elif action == TRAIN:
#        todo
#        # Record several images from webcam of user and save to gallery/system-owner%i.png.
#    elif action == DAEMON:
#        print 'daemon'
#        fk = FaceKey()
#        with daemon.DaemonContext(stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr):
#            print 'daemon ctx'
#            fk.run()
 
if __name__ == "__main__":
    from optparse import OptionParser
    usage = """usage: %%prog [options] <%s>
    
Actions:
    
  install := Installs the script's executable daemon.
    
  restart := Stops and then starts the daemon.
    
  run := Runs the script in non-daemon mode.
    
  start := Runs the script in daemon mode.
    
  stop := Terminates a running daemon.
    
  train := Records user images for training the script to recognize the user.
    
  uninstall := Removes the scripts executable daemon.""" % ('|'.join(ACTIONS),)
    parser = OptionParser(usage=usage, version=__version__)
    
    parser.add_option(
        "--pidfile",
        dest="pidfile",
        default=DEFAULT_PIDFILE,
        help="Location of the process identification file storing the " +
            "process identification number while running as a daemon.")
    
    parser.add_option(
        "--opencv_dir",
        dest="opencv_dir",
        default=OPENCV_DIR,
        help="Location of OpenCV media directory.")
    
    parser.add_option(
        "--cascade",
        dest="cascade",
        default=DEFAULT_CASCADE,
        help="Name of the cascade file to use in the OpenCV media directory.")
    
    parser.add_option(
        "--images_dir",
        dest="images_dir",
        default=DEFAULT_IMAGES_DIR,
        help="Directory where training face images are stored.")

    (options, args) = parser.parse_args()
    
    daemon = FaceKey(**options.__dict__)
    
    action = DEFAULT_ACTION
    if args:
        action = args[0]
        if action not in ACTIONS:
            parser.error("Invalid action: %s" % action)
        
    if action == INSTALL:
        todo
    elif action == RESTART:
        daemon.restart()
    elif action == RUN:
        daemon.run()
    elif action == START:
        daemon.start()
    elif action == STOP:
        daemon.stop()
    elif action == TRAIN:
        todo
    elif action == UNINSTALL:
        todo
    sys.exit(0)
    