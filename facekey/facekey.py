#!/usr/bin/python
"""
2012.3.14 CKS
A simple script to monitor the webcam while the machine is locked, and
automatically unlock when the user's face is detected.
"""
VERSION = (0, 1, 2)
__version__ = '.'.join(map(str, VERSION))

import atexit
import datetime
import csv
import commands
import os
import re
import sys
import shutil
import time
import threading
import traceback
import logging
from collections import namedtuple

LOG = logging.getLogger(__name__)

#import dbus
#import dbus.decorators
#import dbus.mainloop.glib
#dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

#import gobject
# Necessary to prevent Python threads from freezing due to the GIL
# when gobject blocks while making system calls.
#gobject.threads_init()

try:
    import cv
except ImportError, e:
    print(("Unable to import OpenCV Python extensions. Ensure OpenCV is " +
        "installed and is compiled to support Python extensions.\n%s") % (e,))
    sys.exit(1)

from daemon import Daemon
import pyfaces

import constants as c

print_lock = threading.Lock()

def print_(*args):
    print_lock.acquire()
    try:
        print>>sys.stderr, ' '.join(str(_) for _ in args)
    finally:
        print_lock.release()

class FaceKey(Daemon):
    
    def __init__(self,
        gui=False,
        opencv_dir=c.OPENCV_DIR,
        cascade=c.DEFAULT_CASCADE,
        images_dir=c.DEFAULT_IMAGES_DIR,
        image_extensions=c.DEFAULT_IMAGE_EXTENSIONS,
        unknown_threshold=c.DEFAULT_UNKNOWN_THRESHOLD,
        eigen_faces=6,
        eigen_threshold=3,
        *args, **kwargs):
        
        super(FaceKey, self).__init__(*args, **kwargs)
        
        # Reference to an image asynchronously captured from a video stream.
        self._capture_image_fresh = False
        self._capture_image = None
        self._face = None
        self._capture_lock = threading.Lock()
        self.capture_being_identified = False
        self.show_face_rect = True
        
        self.loop = None
        self._cam_thread = None
        atexit.register(self._atexit)
        
        self.unknown_threshold = float(unknown_threshold)
        #self.locked = False
        self.gui = gui
        self.images_dir = images_dir
        
        if isinstance(image_extensions, basestring):
            image_extensions = image_extensions.split(',')
            image_extensions = [_.strip().lower() for _ in image_extensions]
            image_extensions = '|'.join('(?:%s)' % _ for _ in image_extensions) + '$'
        self.image_extensions = re.compile(image_extensions, re.I)
        
        # Configure OpenCV.
        self.target_width = 125
        self.target_height = 150
        self.saved_images = 0
        self.base = c.OPENCV_DIR
        assert os.path.isdir(self.base), \
            "Base OpenCV directory %s does not exist." % (self.base,)
        self.cascade = os.path.join(self.base, cascade)
        assert os.path.isfile(self.cascade), \
            "Face cascade file %s does not exist." % (self.cascade,)
        self.face_cascade = cv.Load(self.cascade)
        
        # Configure PyFaces.
        self.pyf = None
        self.eigen_faces = eigen_faces
        self.eigen_threshold = eigen_threshold
        self.init_pyfaces()
        
        # Configure camera monitoring thread.
        self._monitoring_camera = True
        self._cam_thread = None#threading.Thread(target=self._monitor_cam)
        #self._cam_thread.daemon = True
        self.monitor_cam_running = False
        
        # Configure dbus.
#        bus = self.bus = dbus.SessionBus()
#        try:
#            screensaver = self.screensaver = bus.get_object('org.gnome.ScreenSaver','/org/gnome/ScreenSaver')
#            screensaver.connect_to_signal('ActiveChanged', self._on_screensaver_change)
#        except dbus.DBusException:
#            traceback.print_exc()
#            sys.exit(1)
    
    def init_pyfaces(self):
        self.pyf = pyfaces.PyFaces(
            imgsdir=os.path.join(self.images_dir, 'gallery'),
            egfnum=self.eigen_faces,
            thrsh=self.eigen_threshold,
            extn='png')
    
    @property
    def unlockers(self):
        unlockers_fn = os.path.join(self.images_dir, 'unlockers')
        if os.path.isfile(unlockers_fn):
            return [
                _.strip()
                for _ in open(unlockers_fn, 'r').read().split('\n')
                if _.strip()
            ]
        else:
            return []
    
    @property
    def has_training(self):
        unlockers = self.unlockers
        for fn in os.listdir(os.path.join(self.images_dir, 'gallery')):
            name = re.sub('(?:[^a-zA-Z0-9]+)?[0-9]+\.[a-zA-Z]+', '', fn)
            #print fn, name
            if name in unlockers:
                return True
        return False
    
    def _atexit(self):
        if self._cam_thread:
            self._monitoring_camera = False
        if self.loop:
            self.loop.quit()

    def classify(self, files=None, realtime=False, only_when_locked=False):
        """
        Iterates over each argument and looks up the closest image match,
        """
        print 'classify'
        if realtime:
            self.gui = True
            for _ in self._monitor_cam(only_when_locked=only_when_locked):
                yield _
            return
        
        def classify_image(fqfn):
            matches = self.image_extensions.findall(fqfn)
            if not matches:
                return
            print>>sys.stderr, 'Classifying %s...' % fqfn
            for face in self._iter_face_names(fqfn):
                yield face
        
        for path in files:
            if os.path.isdir(path):
                for dirpath, dirname, files in os.walk(path):
                    for fn in sorted(files):
                        fqfn = os.path.join(dirpath, fn)
                        for _ in classify_image(fqfn):
                            yield _
            elif os.path.isfile(path):
                fqfn = os.path.abspath(path)
                for _ in classify_image(fqfn):
                    yield _

    def collect(self, start_dir, recurse=True):
        """
        Scans a directory for images containing faces.
        """
        faces_found = 0
        good_faces_found = 0
        
        checked_dir = os.path.join(self.images_dir, '_checked')
        if not os.path.isdir(checked_dir):
            os.makedirs(checked_dir)
         
        for dirpath, dirname, files in os.walk(start_dir):
            #print dirpath, dirname, sorted(files)
            #break
            for fn in sorted(files):
                fqfn = os.path.join(*([dirpath] + [fn]))
                checked_key = '%s_%i_%i' % (
                    re.sub('[^a-zA-Z0-9_\-\.]+', '_', fqfn),
                    self.target_width,
                    self.target_height)
                checked_fqfn = os.path.join(checked_dir, checked_key+'.checked')
                if os.path.isfile(checked_fqfn):
                    continue
                open(checked_fqfn, 'w')
                print 'Checking %s...' % (fqfn,)
                
                matches = self.image_extensions.findall(fqfn)
                if not matches:
                    continue
                
                faces, image = self.detect(fqfn)
                
                if faces:
                    print 'Face%s detected!' % ('s' if len(faces) > 1 else '',)
                    check_count = 0
                    for face in faces:
                        check_count += 1
                        faces_found += 1
                        
                        rect,neighbors = face
                        _x, _y, _width, _height = rect
                        height = _height
                        width = cv.Round(self.target_width/float(self.target_height)*height)
                        y = _y
                        x = _x + cv.Round(abs(width-_width)/2.)
                        print '\tat: (%i, %i), size: %ix%i' % (_x, _y, width, height)
                        
                        if width >= self.target_width and height >= self.target_height:
                            good_faces_found += 1
                            
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
                            #image_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
                            image_id = '%s_%i_%i_%i_%i' % (checked_key.lower(), _x, _y, _width, _height)
                            pending_dir = os.path.join(self.images_dir, 'pending')
                            if not os.path.isdir(pending_dir):
                                os.makedirs(pending_dir)
                            _fqfn = os.path.join(pending_dir, "unknown_%s.png" % (image_id,))
                            cv.SaveImage(_fqfn, thumbnail)
                            print '!'*80
                            print 'Saved %s.' % (_fqfn,)
                            assert os.path.isfile(_fqfn)
                        else:
                            print 'Face too small. No smaller than %ix%i but face is %ix%i.' \
                                % (self.target_width, self.target_height, _width, _height)
            if not recurse:
                break
        print '-'*80
        print '%i faces found.' % faces_found
        print '%i good faces found.' % good_faces_found

    def detect(self, image):
        """
        Determines the location of one or more faces in the given image.
        """
        if isinstance(image, basestring):
            image = cv.LoadImage(image)
        
        image_size = cv.GetSize(image)
    
        # Convert to grayscale.
        grayscale = cv.CreateImage(image_size, 8, 1)
        cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)
        #cv.EqualizeHist(grayscale, grayscale)
        _image = image
        image = grayscale
        
        # Assume an image equal to the target size contains a single face.
        target_size = (self.target_width, self.target_height)
        if image_size == target_size:
            rect = 0, 0, self.target_width, self.target_height
            neighbors = None
            return [(rect, neighbors)], image
        
        # Locate one or more faces in the image.
        t0 = time.time()
        faces = cv.HaarDetectObjects(
            image, 
            self.face_cascade,
            cv.CreateMemStorage(0), 1.2, 2, 0, (20, 20))
        t1 = time.time() - t0
        LOG.info('Haar detect took %.2f seconds.' % (t1,))
        return faces, image

    def _iter_face_names(self, image):
        """
        Detects faces in the image and attempts to identify them.
        Returns a generator iterating over all names of identified faces.
        """
        
        fqfn0 = None
        if isinstance(image, basestring):
            fqfn0 = image
        
        Face = namedtuple('Face', ('filename', 'x', 'y', 'width', 'height', 'name', 'dist'))
    
        # Convert to grayscale.
        #image_size = cv.GetSize(image)
        #grayscale = cv.CreateImage(image_size, 8, 1)
        #cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)
        #cv.EqualizeHist(grayscale, grayscale)
        _image = image
        #image = grayscale
        
#        t0 = time.time()
#        faces = cv.HaarDetectObjects(
#            image, 
#            self.face_cascade,
#            cv.CreateMemStorage(0), 1.2, 2, 0, (20, 20))
#        t1 = time.time() - t0
#        print 'haar secs:',t1
        faces, image = self.detect(image)
     
        if faces:
            #print 'Face detected!'
            for face in faces:
                #print_('iter_face_names.face:',face)
                rect, neighbors = face
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
                    print_('Identifying %s...' % (fqfn,))
                    t0 = time.time()
                    name, match_dist = self.pyf.match_name(fqfn)
                    t1 = time.time() - t0
                    #print_('Pyfaces secs:',t1)
                    if name:
                        #print_('-'*80)
                        name = re.sub('[^a-zA-Z0-9]+$', '', name)
                        print_('RECOGNIZED:', name, 'seconds:', t1)
                        face_obj = Face(fqfn0, x, y, width, height, name, match_dist)
                        yield face_obj
                    else:
#                        print '-'*80
                        print_('UNRECOGNIZED', 'seconds:', t1)
                    #os.remove(fqfn)
                else:
                    print_('Ignoring too small face at %i %i %ix%i.' % (x, y, width, height))
                
                # Indicate area containing face.
                #TODO:fix
#                if self.gui:
#                    cv.Rectangle(image, (x,y), (x+width,y+height), cv.RGB(255,0,0))
    
    def set_capture_image(self, image):
        self._capture_lock.acquire()
        try:
            self._capture_image_fresh = True
            self._capture_image = image
        finally:
            self._capture_lock.release()
        
    def get_capture_image(self):
        self._capture_lock.acquire()
        try:
            self._capture_image_fresh = False
            return self._capture_image
        finally:
            self._capture_lock.release()
    
    def set_face(self, face):
        self._capture_lock.acquire()
        try:
            self._face = face
        finally:
            self._capture_lock.release()
        
    def get_face(self):
        self._capture_lock.acquire()
        try:
            return self._face
        finally:
            self._capture_lock.release()
    
    @property
    def capture_image_fresh(self):
        return self._capture_image_fresh
    
    def _monitor_cam(self, only_when_locked=True):
        """
        Attempts to detect and identify faces in a live video stream
        from a locally attached webcam.
        """
        parent_self = self
        self.monitor_cam_running = True
        
        print 'Monitor cam thread started.'
        
        class CamThread(threading.Thread):
            """
            Asynchronously captures web cam frames.
            """
            
            stop = False
            
            def run(self):
                _self = self
                self = parent_self

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
                    print_("Error opening capture device")
                    #sys.exit(1)
                    return
                
                print_('Monitoring camera...')
                print_('Press ESC to exit.')
                delay_ms = 100
                while not _self.stop and self._monitoring_camera:
                    frame = None
                    
                    if only_when_locked and not self.locked:
                        print_('Not locked. Sleeping.')
                        time.sleep(1)
                        continue
             
                    # Capture an image from the camera.
                    frame = cv.QueryFrame(capture)
                    if frame is None:
                        break
                    cv.Flip(frame, None, 1)
             
                    if not self.capture_being_identified:
                        self.set_capture_image(frame)
             
                    # Display webcam image.
                    if self.gui:
                        if frame:
                            face = self.get_face()
                            if face and self.show_face_rect:
                                cv.Rectangle(
                                    frame,
                                    (face.x,face.y),
                                    (face.x+face.width,face.y+face.height),
                                    cv.RGB(255,0,0)
                                )
                            cv.ShowImage('Camera', frame)
                 
                        # handle events
                        k = cv.WaitKey(delay_ms)
                 
                        if k == 0x1b: # ESC
                            print_('ESC pressed. Exiting...')
                            break
                    else:
                        time.sleep(delay_ms*1/1000.)
        
        class DetectThread(threading.Thread):
            """
            Asynchronously detects faces in captured web cam frames.
            """
            
            stop = False
            
            def run(self):
                _self = self
                self = parent_self
                while not _self.stop:
                    if self.capture_image_fresh:
                        self.capture_being_identified = True
                        #print_('Fresh face found!')
                        # Detect and recognize faces.
                        frame = self.get_capture_image()
                        for face in self._iter_face_names(frame):
                            self.set_face(face)
                            print_('Found name:',face)
                            if face.name in self.unlockers:
                                self.unlock()
                        else:
                            print_('No faces found.')
                    else:
                        time.sleep(1)
                    self.capture_being_identified = False
        try:
            cam_thread = CamThread()
            cam_thread.daemon = True
            cam_thread.start()
            
            detect_thread = DetectThread()
            detect_thread.daemon = True
            detect_thread.start()
            
            while 1:
                if not cam_thread.is_alive():
                    break
                time.sleep(1)
                yield
            
            detect_thread.stop = True
            #detect_thread.join()
            
        finally:
            self.monitor_cam_running = False
    
#    def _on_screensaver_change(self, locked):
#        locked_name = 'locked' if locked else 'unlocked'
#        print 'Screensaver changed to %s.' % (locked_name,)
#        self.locked = bool(locked)
#        if self.locked:
#            self.last_datetime_locked = datetime.datetime.now()
#            #TODO:delay autounlock after N seconds
    
    def run(self):
        unlockers_fn = os.path.join(self.images_dir, 'unlockers')
        assert self.unlockers, \
            ("No unlockers are specified. Add names of one or more " +
             "unlockers to %s.") % (unlockers_fn,)
        assert self.has_training, \
            ("No training images detected. Please run `%s train` to train " +
             "the script to recognize your face.") % (__file__)
        self.gui = False
        self.show_face_rect = False
        for _ in self.classify(realtime=True, only_when_locked=True):
            #print 'realtime:',_
            pass
        return
            
#        self._cam_thread = threading.Thread(target=self._monitor_cam)
#        self._cam_thread.daemon = True
#        self._cam_thread.start()
        
#        while 1:
#            if self.monitor_cam_running:
#                break
#            print 'Waiting for monitor cam thread to start...', self._cam_thread.isAlive()
#            time.sleep(1)
            
#        print "Monitoring screensaver..."
#        loop = self.loop = gobject.MainLoop()
#        loop.run()
#        self._cam_thread.join()
    
    @property
    def locked(self):
        output = commands.getoutput('DISPLAY=:0 gnome-screensaver-command -q')
        return not ('inactive' in output)
    
    def lock(self):
        #self.screensaver.SetActive(True)
        output = commands.getoutput('DISPLAY=:0 gnome-screensaver-command -l')
    
    def unlock(self):
        if not self.locked:
            return
        #self.screensaver.SetActive(False)
        output = commands.getoutput('DISPLAY=:0 gnome-screensaver-command -d')

if __name__ == "__main__":
    from optparse import OptionParser
    usage = """usage: %%prog [options] <%s>
    
Actions:
    
    classify := Detect and identify faces in one or more images.
    
    collect := Search for faces in images, and extract as a training sample.
    
    install := Installs the script's executable daemon.
    
    restart := Stops and then starts the daemon.
    
    run := Runs the script in non-daemon mode.
    
    start := Runs the script in daemon mode.
    
    stop := Terminates a running daemon.
    
    train := Records user images for training the script to recognize the user.
    
    uninstall := Removes the scripts executable daemon.""" \
        % ('|'.join(c.ACTIONS),)
    parser = OptionParser(usage=usage, version=__version__)
    
    parser.add_option(
        "--pidfile",
        dest="pidfile",
        default=c.DEFAULT_PIDFILE,
        help='''Location of the process identification file storing the
            process identification number while running as a daemon.''')
    
    parser.add_option(
        "--opencv_dir",
        dest="opencv_dir",
        default=c.OPENCV_DIR,
        help="Location of OpenCV media directory.")
    
    parser.add_option(
        "--cascade",
        dest="cascade",
        default=c.DEFAULT_CASCADE,
        help="Name of the cascade file to use in the OpenCV media directory.")
    
    parser.add_option(
        "--images_dir",
        dest="images_dir",
        default=c.DEFAULT_IMAGES_DIR,
        help="Directory where training face images are stored.")
    
    parser.add_option(
        "--image_extensions",
        default='.jpg,.png',
        help="Directory where training face images are stored.")
    
    parser.add_option(
        "--no_recurse",
        action="store_true",
        default=False,
        help="Indicates file processing should be not be recursive.")
    
    parser.add_option(
        "--realtime",
        action="store_true",
        default=False,
        help='''Indicates classification should be performed on
            a live video feed.''')
    
    parser.add_option(
        "--unknown_threshold",
        default=c.DEFAULT_UNKNOWN_THRESHOLD,
        help='''The match distance over which an image will be classified
            as "unknown".''')

    (options, args) = parser.parse_args()
    
    daemon = FaceKey(**options.__dict__)
    
    action = c.DEFAULT_ACTION
    if args:
        action = args[0]
        if action not in c.ACTIONS:
            parser.error("Invalid action: %s" % action)
        
    if action == c.INSTALL:
        todo
    elif action == c.CLEAN:
        #shutil.rmtree(os.path.join(c.DEFAULT_IMAGES_DIR, 'probes', '.*'))
        del_path = os.path.join(c.DEFAULT_IMAGES_DIR, 'probes', '*')
        del_cmd = 'rm -Rf %s' % (del_path,)
        #print del_cmd
        os.system(del_cmd)
    elif action == c.RESTART:
        daemon.restart()
    elif action == c.RUN:
        daemon.run()
    elif action == c.START:
        daemon.start()
    elif action == c.STOP:
        daemon.stop()
    elif action == c.COLLECT:
        daemon.collect(start_dir=args[1], recurse=not options.no_recurse)
    elif action == c.TRAIN:
        print 'Training...'
        daemon.pyf.train()
    elif action == c.UNINSTALL:
        todo
    elif action == c.CLASSIFY:
        realtime = options.realtime
        if realtime:
            #print 'realtime'
            for _ in daemon.classify(realtime=True):
                #print 'realtime:',_
                pass
        else:
            headers='filename,x,y,width,height,name,dist'.split(',')
            print>>sys.stdout, ','.join(headers)
            dw = csv.DictWriter(sys.stdout, headers)
            for _ in daemon.classify(files=args[1:]):
                name = _.name
    #            print _.dist, daemon.unknown_threshold, _.dist > daemon.unknown_threshold
                if _.dist > daemon.unknown_threshold:
                    name = c.UNKNOWN
                dw.writerow(dict(zip(
                    headers,
                    (_.filename, _.x, _.y, _.width, _.height, name, _.dist))))
                
        sys.exit(0)
        