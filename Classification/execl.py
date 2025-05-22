import os


def extract_log_files(input_folder, output_file):
    # 存储已经写入过的文件名的集合
    written_files = set()

    # 检查输出文件中已经写入过的文件名
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as vtab_file:
            for line in vtab_file:
                file_name = line.split()[0]
                written_files.add(file_name)

    # 打开vtab.txt文件以便写入
    with open(output_file, 'a', encoding='utf-8') as vtab_file:
        # 遍历文件夹中的文件
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.endswith('.log') and file not in written_files:
                    # 构建完整的文件路径
                    file_path = os.path.join(root, file)

                    # 打开.log文件并提取文字
                    with open(file_path, 'r', encoding='utf-8') as log_file:
                        log = log_file.read()
                        log_text = log
                        # 将提取的文字写入vtab.txtv
                        vtab_file.write(f"{file} {log_text}\n")
                    written_files.add(file)
