import re
string = open("./myoutput_final.txt","r+").read()
split_str = string.split('\n0 / 134')
#print(split_str)
# Extract the step numbers from the string
for trail_idx, s in enumerate(split_str):
    steps = re.findall(r'\d+>', s)
    len_trial = 10 + 5 * (2 * trail_idx // 3)
    fail = re.findall(r'{len_trial}>', s)
    print(len(steps))
    print(steps)