import psutil
import os
info = psutil.virtual_memory()
print('内存使用：',psutil.Process(os.getpid()).memory_info().rss)
print('总内存：',info.total)
print('内存占比：',info.percent)
print('CPU个数：',psutil.cpu_count())
