#!/bin/bash

# 检查是否提供了文件名作为参数
if [ $# -eq 0 ]; then
    echo "请提供文件名作为参数。"
    exit 1
fi

# 获取输入的文件名
file="$1"

# 检查文件是否存在
if [ ! -f "$file" ]; then
    echo "文件 $file 不存在。"
    exit 1
fi

count=0
while read -r line; do
    # 使用正则表达式提取 Testing accuracy 后面的数值
    accuracy=$(echo "$line" | grep -oP '(?<=Testing accuracy: )\d+\.?\d*')
    if [ -n "$accuracy" ]; then
        if [ $((count % 15)) -eq 0 ]; then
            echo "($count,$accuracy)"
        fi
        ((count++))
    fi
done < "$file"