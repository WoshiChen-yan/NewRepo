# 删除 test 目录下常见图片文件
import os, glob

root = "test"
exts = ("*.png","*.jpg","*.jpeg","*.gif","*.csv")
count = 0
for e in exts:
    for p in glob.glob(os.path.join(root, "**", e), recursive=True):
        try:
            os.remove(p)
            count += 1
            print("Deleted:", p)
        except Exception as ex:
            print("Failed:", p, ex)
print(f"Total deleted: {count}")