import ctypes
import time
from ctypes import windll


class Win32ApiMouseMover:
    def __init__(self, move_step=1, move_frequency=0.001):
        self.intention = None
        self.change_coordinates_num = 0
        self.user32 = windll.user32
        self.move_step = move_step
        self.move_frequency = move_frequency

    def set_intention(self, move_x, move_y):
        """设置移动意图"""
        self.intention = move_x, move_y
        self.change_coordinates_num += 1

    def start(self):
        timer = Timer()
        """启动鼠标移动器"""
        print("win32api鼠标移动器启动")
        while True:
            if self.intention is not None:
                t0 = time.time()
                (x, y) = self.intention
                print("开始移动，移动距离:{}".format((x, y)))
                while x != 0 or y != 0:
                    (x, y) = self.intention
                    move_up = min(self.move_step, abs(x)) * (1 if x > 0 else -1)
                    move_down = min(self.move_step, abs(y)) * (1 if y > 0 else -1)
                    if x == 0:
                        move_up = 0
                    elif y == 0:
                        move_down = 0
                    x -= move_up
                    y -= move_down
                    self.intention = (x, y)
                    self.user32.mouse_event(0x1, int(move_up), int(move_down))
                    timer.sleep(0.001)
                print(
                    "完成移动时间:{:.2f}ms,坐标变更次数:{}".format((time.time() - t0) * 1000,
                                                                   self.change_coordinates_num))
            self.intention = None
            self.change_coordinates_num = 0
            time.sleep(0.001)


class Timer(object):

    def __init__(self):
        freq = ctypes.c_longlong(0)
        ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(freq))
        self.__freq = freq.value
        self.__beginCount = self.counter()

    def counter(self):
        freq = ctypes.c_longlong(0)
        ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(freq))
        return freq.value

    def beginCount(self):
        self.__beginCount = self.counter()

    # 时间差，精确到微秒
    def secondsDiff(self):
        self.__endCount = self.counter()
        return (self.__endCount - self.__beginCount) / (self.__freq + 0.)

    # 休眠，精确到毫秒
    def sleep(self, timeout):
        while True:
            self.__endCount = self.counter()
            if ((self.__endCount - self.__beginCount) / (self.__freq + 0.)) * 1000 >= timeout:
                return
