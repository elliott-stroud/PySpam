import datetime

from inspect import currentframe, getframeinfo


def log(msg=None, level='LOG'):
    fi = getframeinfo(currentframe().f_back)

    print("[{level}][{time}][{file}: Line #{line}]".format(
        level=level,
        file=fi.filename,
        line=fi.lineno,
        #time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Adding milliseconds for better timing for Algorithm speed
        time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    ), msg)
