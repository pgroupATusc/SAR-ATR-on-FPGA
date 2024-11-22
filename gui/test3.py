from subprocess import run
import re
import time
#data = run("sensors | grep POWER | head -n 1",capture_output=True,shell=True)
#print(data.stdout)

while True:
    f = open("cpupower.text", "w")
    data = run("sudo turbostat --Summary --quiet --show PkgWatt --interval 0.1 --num_iterations 1",capture_output=True,shell=True)
    powerinfo = str(data.stdout)
    powerinfo = re.findall(r'\b\d+\b',powerinfo)
    powerinfo = powerinfo[0]
    f.write(powerinfo)
    f.close()
    
    time.sleep(0.2)

