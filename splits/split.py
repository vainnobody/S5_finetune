from pathlib import Path

class SplitReeader:
    def __init__(self, txt_path):
        self.txt_path = txt_path
        with open(self.txt_path, 'r') as f:
            self.ids = f.readlines()
    
    def save(self, save_path):
        with open(save_path, 'w', encoding='utf-8') as f:
            f.writelines(self.ids)

    def test(self, func):
        id = self.ids[0]
        print(f"pre: {id}")
        id = func(id)
        print(f"post: {id}")

    def __call__(self, func):
        for i in range(len(self.ids)):
            self.ids[i] = func(self.ids[i])

    def __str__(self):
        return str(self.ids)

def modified_func(id):
    path1, path2 = id.split()
    path1 = Path(path1).relative_to("labeled")
    path2 = Path(path2).relative_to("labeled")
    new_id = f"{str(path1)} {str(path2)}\n"
    return new_id


txt_path = '/data/users/lanjie/Project/S5_finetune/splits/IRSAMap/val.txt'
save_path = '/data/users/lanjie/Project/S5_finetune/splits/IRSAMap/val_wo_labeled.txt'
reader = SplitReeader(txt_path)
reader.test(modified_func)
reader(modified_func)
reader.save(save_path)
# print(reader)