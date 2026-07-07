#import corepy.convop as convop
import corepy as core
import mbp16dpy as mlib

src = mlib.matrixbp(4,4)
src.fromlist([i for i in range(16)])
print(src)
to = mlib.matrixbp(2,2)
core.convop.averpool(src,(2,2),2,to)

print(to)