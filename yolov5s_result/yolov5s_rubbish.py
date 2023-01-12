from ax import pipeline
import time
import threading

def pipeline_data(threadName, delay):
    time.sleep(0.2) # wait for pipeline.work() is True
    for i in range(200):
        time.sleep(delay)
        tmp = pipeline.result()
        if tmp and tmp['nObjSize']:
            for i in tmp['mObjects']:
                print(i)
    pipeline.free() # 400 * 0.05s auto exit pipeline

thread = threading.Thread(target=pipeline_data, args=("Thread-1", 0.05, ))
thread.start()

pipeline.load([
    b'libsample_vin_ivps_joint_vo_sipy.so',
    b'-p', b'/home/yolov5s_rubbish.json',
    b'-c', b'2',
])

thread.join() # wait thread exit
