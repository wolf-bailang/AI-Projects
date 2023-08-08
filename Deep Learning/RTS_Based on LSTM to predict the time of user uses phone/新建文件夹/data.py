import math

filename = 'data.txt'      # txt文件和当前脚本在同一目录下，所以不用写具体路径
f1 = open('out.txt', 'w')

with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readlines()      # 整行读取数据
        if not lines:
            break
        for i in range(0, len(lines)):
            lines[i] = lines[i].rstrip('\n')
            if (i % 2) == 0:
                pos = []
                pos = '[' + lines[i] + ']' 
                f1.writelines(pos)
            else:
                pos = []
                pos = '(' + lines[i] + ')' + '\n'
                f1.writelines(pos)
        print('end')
        f1.close()

