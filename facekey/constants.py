import os

ACTIONS = (
    CLASSIFY,
    CLEAN,
    COLLECT,
    INSTALL,
    UNINSTALL,
    TRAIN,
    START,
    STOP,
    RESTART,
    RUN,
) = (
    'classify',
    'clean',
    'collect',
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

OPENCV_DIR = os.environ.get(
    'FACEKEY_OPENCV_DIR', '/usr/local/share/OpenCV')
DEFAULT_CASCADE = os.environ.get(
    'FACEKEY_FACE_HAAR', 'haarcascades/haarcascade_frontalface_alt.xml')
DEFAULT_IMAGES_DIR = os.environ.get(
    #'FACEKEY_IMAGES_DIR', os.path.join(os.path.dirname(__file__), 'images'))
    'FACEKEY_IMAGES_DIR', os.path.join(os.path.expanduser('~'), '.facekey'))

DEFAULT_IMAGE_EXTENSIONS = '.jpg,*.jpeg,.png'

DEFAULT_UNKNOWN_THRESHOLD = 0.07

UNKNOWN = 'UNKNOWN'
