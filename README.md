
<h1 align="center"><b>CS331.M21.KHCL</b></h>

# THÀNH VIÊN NHÓM
| STT    | MSSV          | Họ và Tên              |Chức Vụ    | Email                   |
| ------ |:-------------:| ----------------------:|----------:|-------------------------:
| 1      | 19521242      | Lương Phạm Bảo         |Nhóm trưởng |19521242@gm.uit.edu.vn   |
| 2      | 19520993      | Nguyễn Gia Thống       |Thành viên  |19520993@gm.uit.edu.vn   |
| 3      | 19521412      | Phạm Ngọc Dương        |Thành viên  |19521412@gm.uit.edu.vn   |

# GIỚI THIỆU MÔN HỌC
* **Tên môn học:** Thị giác máy tính nâng cao
* **Mã lớp:** CS331.M21.KHCL
* **Năm học:** HK2 (2021 - 2022).
* **Giảng viên**: Ts. Nguyễn Vĩnh Tiệp

# GIỚI THIỆU ĐỒ ÁN
* **Tên đồ án:** ClipCap: Fast and Simple Image Captioning model using CLIP & GPT-2 

## How to run
1. Install require packages and download datasets, pretrained embedding,trained weights.
```
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
./sh down_viecap.sh
./sh down_uitvic.sh
/sh  weights.sh
```
3. Training, we using  TPU from Colab PRO (with at least 28 GB RAM) for faster training.

```
# accelerate launch --config_file ./config_tpu.yml train.py --batch-size=32 

## Acknowledgments

We would like to thank Thanh V. Le and  GPT2 team for providing valuable resources from VLSP 2021 viecap4h Challenge
4. Inference (run on cuda devices), you can follow [this notebook](infer.ipynb).

## References

- OpenAI [CLIP](https://openai.com/blog/clip/)
- rmokady et al - [CLIPCap](https://github.com/rmokady/CLIP_prefix_caption)
- [GPT2News - imthanhlv](https://huggingface.co/imthanhlv/gpt2news)
- GPT2 Team solution (https://github.com/Luvata/vlsp_viecap4h_gptteam_code)

