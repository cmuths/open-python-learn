import colorsys

# RGB颜色值（0-255）
r, g, b = 100, 70, 70

# 将RGB值归一化到[0, 1]范围
r, g, b = r / 255.0, g / 255.0, b / 255.0

# 计算H、S、V的值
h, s, v = colorsys.rgb_to_hsv(r, g, b)

# 输出HSV值
print("H:", h)
print("S:", s)
print("V:", v)