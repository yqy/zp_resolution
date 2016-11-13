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
        print "Turn ",echo_num

        zp_num = 0
        hits = 0
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
                    line = line.split("\t")
                    items = []
                    if len(line) == 4:
                        items.append((line[1],line[2],float(line[3].split(":")[1]))) 

                    this_score = 0.0
                    this_tag = "0"
                    for tag,word,score in items:
                        if score >= this_score:
                            this_score = score
                            this_tag = tag

                    if this_tag == "1":
                        hits += 1
            if line.endswith("seconds!"):
                print zp_num,hits
                ### out ###
                break


print echo_num
