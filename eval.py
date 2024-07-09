import numpy as np
import json
import os
with open("result/video_results_test.json","r") as f:
    video_results=json.load(f)

anno_root="datasets/sensecrowd/new_annotations"


gt_video_num_list=[]
gt_video_len_list=[]
pred_video_num_list=[]
pred_matched_num_list=[]
gt_matched_num_list=[]
for video_name in video_results:
    video_len=0
    anno_path=os.path.join(anno_root,video_name+".txt")
    with open(anno_path,"r") as f:
        lines=f.readlines()
        all_ids=set()

        for line in lines:
            line=line.strip().split(" ")
            data=[float(x) for x in line[3:] if x!=""]
            if len(data)>0:
                data=np.array(data)
                data=np.reshape(data,(-1,7))
                ids=data[:,6].reshape(-1,1)
                for id in ids:
                    all_ids.add(int(id[0]))
    info=video_results[video_name]
    gt_video_num=len(all_ids)
    pred_video_num=info["video_num"]
    pred_video_num_list.append(pred_video_num)
    gt_video_num_list.append(gt_video_num)
    gt_video_len_list.append(info["frame_num"])

        

MAE=np.mean(np.abs(np.array(gt_video_num_list)-np.array(pred_video_num_list)))
MSE=np.mean(np.square(np.array(gt_video_num_list)-np.array(pred_video_num_list)))
WRAE=np.sum(np.abs(np.array(gt_video_num_list)-np.array(pred_video_num_list))*np.array(gt_video_len_list)/np.array(gt_video_num_list)/np.sum(gt_video_len_list))
RMSE=np.sqrt(MSE)

print(f"MAE:{MAE:.2f}, MSE:{MSE:.2f}, WRAE:{WRAE*100:.2f}%, RMSE:{RMSE:.2f}")