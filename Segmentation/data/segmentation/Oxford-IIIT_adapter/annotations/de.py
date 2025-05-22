# 打开源文件和目标文件
with open('test.txt', 'r') as source_file, open('test_1.txt', 'w') as target_file:
    # 读取源文件的每一行
    for line in source_file:
        # 使用split()函数分割每一行，得到第一个空格前的内容
        first_part = line.split()[0]
        # 将这部分内容写入目标文件
        target_file.write(first_part + '\n')