import os
import matplotlib.pyplot as plt
import random
import numpy as np
import math

target_path = '/root/attack/result'

if __name__ == '__main__':
    match_detail = []
    deletebug = 0
    for root, dirs, files in os.walk(target_path):
        for file in files:
            if file.endswith(".txt"):
                fn = os.path.join(root, file)
                with open(fn) as f:
                    lines = f.readlines()
                    dic = {}
                    match_detail.append(dic)
                    dic["filename"] = fn.split("/")[-1]
                    dic["state"] = "fail"
                    dic["state_detail"] = "default"

                    dic["funcA"] = eval(lines[0])
                    dic["funcB"] = eval(lines[1])

                    dic["bug"] = 0
                    
                    for i in range(len(lines)):
                        lines[i] = lines[i].strip()
                        if i <= 1:
                            continue
                        if "delete" in lines[i] and i == 2:
                            deletebug += 1
                            dic["bug"] = 1
                        if lines[i].startswith("success"):
                            dic["state"] = "success"
                            dic["state_detail"] = "success"
                        if lines[i].endswith("no-solution"):
                            dic["state_detail"] = "floyd_fail"
                        if lines[i].endswith("candidates"):
                            dic["state_detail"] = "candidate_fail"
                        if lines[i].endswith("solution-now"):
                            dic["state_detail"] = "cannotfound_fail"
                        if lines[i].startswith("using"):
                            t = float(lines[i][5:-2])
                            if(t > 60) :
                                t = -0.2
                            dic["time"] = t
                    
                    if dic["state_detail"] == "default":
                        dic["state_detail"] = "tle_fail"

    tle_f = open("0_tle_match.txt", "w+")
    bug_f = open("0_bug_match.txt", "w+")
    for match in match_detail :
        if match["state_detail"] == "tle_fail":
            print( match, file = tle_f)
        if match["bug"] == 1:
            print( match, file = bug_f)
    tle_f.close()
    bug_f.close()


    exit()

    count = 0
    fail_floyd_count = 0
    fail_candidate_count = 0
    fail_cannotfound_count = 0
    fail_tle_count = 0
    success_count = 0

    fail_floyd_x = []
    fail_floyd_y = []

    fail_candidate_x = []
    fail_candidate_y = []

    fail_cannotfound_x = []
    fail_cannotfound_y = []

    tle_x = []
    tle_y = []

    success_x = []
    success_y = []

    time_success_x = []
    time_success_y = []

    time_floyd_x = []
    time_floyd_y = []

    time_cannotfound_x = []
    time_cannotfound_y = []

    time_candidate_x = []
    time_candidate_y = []

    count = len(match_detail)
    for match in match_detail :
        if match["state_detail"] == "candidate_fail":
            fail_candidate_count+=1
            fail_candidate_x.append(int(match["funcA"]["bbs"])+ random.random() - 0.5)
            fail_candidate_y.append(int(match["funcB"]["bbs"])+ random.random() - 0.5)
            time_candidate_y.append(match["time"])
            time_candidate_x.append(math.log(int(match["funcA"]["bbs"])) + math.log(int(match["funcA"]["bbs"])) + random.random() - 0.5)
        if match["state_detail"] == "cannotfound_fail":
            fail_cannotfound_count += 1
            fail_cannotfound_x.append(int(match["funcA"]["bbs"])+ random.random() - 0.5)
            fail_cannotfound_y.append(int(match["funcB"]["bbs"])+ random.random() - 0.5)
            time_cannotfound_y.append(match["time"])
            time_cannotfound_x.append(math.log(int(match["funcA"]["bbs"])) + math.log(int(match["funcA"]["bbs"])) + random.random() - 0.5)
        if match["state_detail"] == "floyd_fail":
            fail_floyd_count += 1
            fail_floyd_x.append(int(match["funcA"]["bbs"])+ random.random() - 0.5)
            fail_floyd_y.append(int(match["funcB"]["bbs"])+ random.random() - 0.5)
            time_floyd_y.append(match["time"])
            time_floyd_x.append(math.log(int(match["funcA"]["bbs"])) + math.log(int(match["funcA"]["bbs"])) + random.random() - 0.5)
        if match["state_detail"] == "tle_fail":
            fail_tle_count += 1
            tle_x.append(int(match["funcA"]["bbs"])+ random.random() - 0.5)
            tle_y.append(int(match["funcB"]["bbs"])+ random.random() - 0.5)
        if match["state"] == "success":
            success_count += 1
            success_x.append(int(match["funcA"]["bbs"])+ random.random() - 0.5)
            success_y.append(int(match["funcB"]["bbs"]) + random.random() - 0.5)
            time_success_y.append(match["time"])
            time_success_x.append(math.log(int(match["funcA"]["bbs"])) + math.log(int(match["funcA"]["bbs"])) + random.random() - 0.5)
    
    print("success:", success_count/count)
    print("fail:", (fail_cannotfound_count + fail_candidate_count + fail_floyd_count) / count)
    print("Time Limit Exceeded", fail_tle_count/count)
    print("fail_cannotfound:", fail_cannotfound_count/count)
    print("fail_candidate:", fail_candidate_count/count)
    print("fail_floyd:", fail_floyd_count/count)

    print("\nbug:",deletebug/count)

    plt.rcParams['savefig.dpi'] = 500
    plt.scatter(success_x, success_y, color = "red", s = 0.2, label="success")
    plt.scatter(fail_cannotfound_x, fail_cannotfound_y, s = 0.2, color = "blue", label="cannotfound")
    plt.scatter(fail_candidate_x, fail_candidate_y, s = 0.2, color = "green", label="nocandidate")
    plt.scatter(fail_floyd_x, fail_floyd_y, s = 0.2, color = "darkslategrey", label="nofloyd")
    plt.scatter(tle_x, tle_y, color="lightskyblue", s = 0.2, label="time limit exceeded")
    plt.xlabel('funcA BBs')
    plt.ylabel('funcB BBs')
    plt.title("match")
    plt.legend()
    plt.savefig('./0_match.png')
    plt.close()

    plt.rcParams['savefig.dpi'] = 500
    plt.scatter(fail_cannotfound_x, fail_cannotfound_y, s = 0.2, color = "green", label="cannotfound")
    plt.scatter(fail_candidate_x, fail_candidate_y, s = 0.2, color = "green", label="nocandidate")
    plt.scatter(fail_floyd_x, fail_floyd_y, s = 0.2, color = "darkslategrey", label="nofloyd")
    plt.scatter(success_x, success_y, color = "red", s = 0.2, label="success")
    plt.xlabel('funcA BBs')
    plt.ylabel('funcB BBs')
    plt.title("match")
    plt.legend()
    plt.savefig('./0_success_and_fail.png')
    plt.close()

    plt.rcParams['savefig.dpi'] = 500
    plt.scatter(fail_candidate_x, fail_candidate_y, s = 0.2, color = "green", label="nocandidate")
    plt.scatter(fail_floyd_x, fail_floyd_y, s = 0.2, color = "red", label="nofloyd")
    plt.scatter(fail_cannotfound_x, fail_cannotfound_y, s = 0.2, color = "blue", label="cannotfound")
    plt.xlabel('funcA BBs')
    plt.ylabel('funcB BBs')
    plt.title("match")
    plt.legend()
    plt.savefig('./0_fail.png')
    plt.close()

    plt.rcParams['savefig.dpi'] = 500
    plt.scatter(success_x, success_y, color = "red", s = 0.2, label="success")
    plt.xlabel('funcA BBs')
    plt.ylabel('funcB BBs')
    plt.title("match")
    plt.legend()
    plt.savefig('./0_success.png')
    plt.close()

    plt.rcParams['savefig.dpi'] = 500
    plt.scatter(tle_x, tle_y, color="blue", s = 0.2, label="time limit exceeded")
    plt.xlabel('funcA BBs')
    plt.ylabel('funcB BBs')
    plt.title("match")
    plt.legend()
    plt.savefig('./0_tle.png')
    plt.close()

    plt.rcParams['savefig.dpi'] = 500
    plt.scatter(fail_candidate_x, fail_candidate_y, s = 0.2, color = "green", label="nocandidate")
    plt.xlabel('funcA BBs')
    plt.ylabel('funcB BBs')
    plt.title("match")
    plt.legend()
    plt.savefig('./0_fail_candidate.png')
    plt.close()

    plt.rcParams['savefig.dpi'] = 500
    plt.scatter(fail_floyd_x, fail_floyd_y, s = 0.2, color = "purple", label="nofloyd")
    plt.xlabel('funcA BBs')
    plt.ylabel('funcB BBs')
    plt.title("match")
    plt.legend()
    plt.savefig('./0_fail_floyd.png')
    plt.close()

    plt.rcParams['savefig.dpi'] = 500
    plt.scatter(fail_cannotfound_x, fail_cannotfound_y, s = 0.2, color = "crimson", label="cannotfound")
    plt.xlabel('funcA BBs')
    plt.ylabel('funcB BBs')
    plt.title("match")
    plt.legend()
    plt.savefig('./0_fail_cannotfound.png')
    plt.close()

    plt.rcParams['savefig.dpi'] = 500
    plt.scatter(time_success_x, time_success_y, s = 0.2, color = "deeppink", label="time_success")
    plt.xlabel('log(funcA BBs * funcBBs)')
    plt.ylabel('time')
    plt.title("match")
    plt.legend()
    plt.savefig('./0_time_success.png')
    plt.close()

    plt.rcParams['savefig.dpi'] = 500
    plt.scatter(time_floyd_x, time_floyd_y, s = 0.2, color = "deeppink", label="time_floyd")
    plt.xlabel('log(funcA BBs * funcBBs)')
    plt.ylabel('time')
    plt.title("match")
    plt.legend()
    plt.savefig('./0_time_floyd.png')
    plt.close()

    plt.rcParams['savefig.dpi'] = 500
    plt.scatter(time_candidate_x, time_candidate_y, s = 0.2, color = "deeppink", label="time_candidate")
    plt.xlabel('log(funcA BBs * funcBBs)')
    plt.ylabel('time')
    plt.title("match")
    plt.legend()
    plt.savefig('./0_time_candidate.png')
    plt.close()



    plt.rcParams['savefig.dpi'] = 500
    plt.scatter(time_cannotfound_x, time_cannotfound_y, s = 0.2, color = "deeppink", label="time_cannotfound")
    plt.xlabel('log(funcA BBs * funcBBs)')
    plt.ylabel('time')
    plt.title("match")
    plt.legend()
    plt.savefig('./0_time_cannotfound.png')
    plt.close()




