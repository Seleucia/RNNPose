import cv2
print(cv2.__version__)

fl='/mnt/Data1/hc/rgb/S1/Videos/Directions.55011271.mp4'
sv='frms/'
vidcap = cv2.VideoCapture('/mnt/Data1/hc/rgb/S1/Videos/Directions.55011271.mp4')
vidcap = cv2.VideoCapture('/home/coskun/PycharmProjects/RNNPoseV2/pred/3.6m/output.avi')
success,image = vidcap.read()
print vidcap.isOpened()
import os, sys
from PIL import Image


# a, b, c = os.popen3("ffmpeg -i "+fl)
# out = c.read()
# dp = out.index("Duration: ")
# duration = out[dp+10:dp+out[dp:].index(",")]
# hh, mm, ss = map(float, duration.split(":"))
# total = (hh*60 + mm)*60 + ss
# for i in xrange(100):
#     # t = (i + 1) * 100 / 10
#     # os.system("ffmpeg -i test.avi -ss %0.3fs frame%i.png" % (t, i))
#     cmd="ffmpeg -i %s -vcodec png -ss %i -vframes 1 -an -f rawvideo frms/frame%i.png" % (fl,i, i)
#     # print(cmd)
#     os.system(cmd)
#     # ffmpeg -i test.avi -vcodec png -ss 10 -vframes 1 -an -f rawvideo test.png
#




count = 0
success = True
while success:
  success,image = vidcap.read()
  print 'Read a new frame: ', success
  cv2.imwrite("frms/frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1