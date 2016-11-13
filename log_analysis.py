import sys

log_file = open(sys.argv[1])
echo_num = 0
while True:
    line = log_file.readline()
    if not line:break
    line = line.strip()
    if line == "Begin test":
        ## new test ##
        echo_num += 1
        print "Turn,"echo_num

        zp_num = 0
        while True:
            line = log_file.readline()
            if not line:break
            line = line.strip()
            if line.startswith("------"):
                zp_num += 1
                while True:
                    line = log_file.readline()
                    if not line:break
                    line = line.strip()
                    if line.startswith("Done ZP"):
                        break 

            if line.endswith("seconds!"):
                print zp_num
                ### out ###
                break


print echo_num
