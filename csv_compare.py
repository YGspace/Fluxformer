import csv

def csv_to_list(csv_path):
    f = open(csv_path,'r')
    rdr = csv.reader(f)
    vid_list = []
    for line in rdr:
        vid_list.append(line[0])
    del vid_list[0]
    #print(len(vid_list))
    #deduped_list = list(set(vid_list))
    #print(len(deduped_list))
    f.close()
    return sorted(list(set(vid_list)))

csv1 = '/home/kimminju/workspace/vid_to_img/trim1.csv'
csv2 = '/home/kimminju/workspace/vid_to_img/trim2.csv'
csv3 = '/home/kimminju/workspace/vid_to_img/failure.csv'
csv4 = '/home/kimminju/workspace/vid_to_img/failure2.csv'
csv5 = '/home/kimminju/workspace/vid_to_img/ori.csv'

trim1 = csv_to_list(csv1)
trim2 = csv_to_list(csv2)
failure = csv_to_list(csv3)
failure2 = csv_to_list(csv4)
ori = csv_to_list(csv5)

print(len(failure), len(failure2), len(ori))
s_fa = set(failure)
s_fa2 = set(failure2)
s_ori = set(ori)
print(len(list(s_fa.intersection(ori))))
# print(trim1[0], trim1[1], trim1[2], trim1[3], trim1[4])
# print(trim2[0], trim2[1], trim2[2], trim2[3], trim2[4])