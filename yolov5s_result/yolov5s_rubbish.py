import time
from ax import pipeline
pipeline.load([
    'libsample_vin_ivps_joint_vo_sipy.so',
    '-p', '/home/yolov5s_rubbish.json',
    '-c', '2',
])
while pipeline.work():
    time.sleep(0.001)
    tmp = pipeline.result()
    if tmp and tmp['nObjSize']:
        for i in tmp['mObjects']:
            print(i)
        # if tmp['nObjSize'] > 10: # try exit
        #     pipeline.free()
pipeline.free()
