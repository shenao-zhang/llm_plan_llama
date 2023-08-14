import re
string = open("./myoutput_remaining.txt","r+").read()
split_str = string.split('\n0 / 134')
cumulative = 3868
# Extract the step numbers from the string
for trail_idx, s in enumerate(split_str):
    steps = re.findall(r'\d+>', s)
    fail = re.findall(r'15>', s)
    success_steps = len(steps) - len(fail) * 16
   # cumulative += success_steps // 2 * (trail_idx**2 + trail_idx)
    tt = trail_idx + 9
    cumulative += success_steps // 2 * (tt**2 + tt)
    print(cumulative, tt)
