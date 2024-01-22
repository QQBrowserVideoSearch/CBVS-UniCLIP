"""
This script provides templates for manual prompting for zero-shot image classification.
"""


openai_templates = [
    lambda c: f"{c}的照片",
    lambda c: f"质量差的{c}的照片",
    lambda c: f"许多{c}的照片",
    lambda c: f"{c}的雕塑",
    lambda c: f"难以看到{c}的照片",
    lambda c: f"{c}的低分辨率照片",
    lambda c: f"{c}的渲染",
    lambda c: f"涂鸦{c}",
    lambda c: f"{c}的糟糕照片",
    lambda c: f"{c}的裁剪照片",
    lambda c: f"{c}的纹身",
    lambda c: f"{c}的刺绣照片",
    lambda c: f"很难看到{c}的照片",
    lambda c: f"{c}的明亮照片",
    lambda c: f"一张干净的{c}的照片",
    lambda c: f"一张包含{c}的照片",
    lambda c: f"{c}的深色照片",
    lambda c: f"{c}的手绘画",
    lambda c: f"我的{c}的照片",
    lambda c: f"不自然的{c}的照片",
    lambda c: f"一张酷的{c}的照片",
    lambda c: f"{c}的特写照片",
    lambda c: f"{c}的黑白照片",
    lambda c: f"一幅{c}的画",
    lambda c: f"一幅{c}的绘画",
    lambda c: f"一张{c}的像素照片",
    lambda c: f"{c}的雕像",
    lambda c: f"一张{c}的明亮照片",
    lambda c: f"{c}的裁剪照片",
    lambda c: f"人造的{c}的照片",
    lambda c: f"一张关于{c}的照片",
    lambda c: f"损坏的{c}的jpeg照片",
    lambda c: f"{c}的模糊照片",
    lambda c: f"{c}的相片",
    lambda c: f"一张{c}的好照片",
    lambda c: f"{c}的渲染照",
    lambda c: f"视频游戏中的{c}",
    lambda c: f"一张{c}的照片",
    lambda c: f"{c}的涂鸦",
    lambda c: f"{c}的近距离照片",
    lambda c: f"{c}的折纸",
    lambda c: f"{c}在视频游戏中",
    lambda c: f"{c}的草图",
    lambda c: f"{c}的涂鸦照",
    lambda c: f"{c}的折纸形状",
    lambda c: f"低分辨率的{c}的照片",
    lambda c: f"玩具{c}",
    lambda c: f"{c}的副本",
    lambda c: f"{c}的干净的照片",
    lambda c: f"一张大{c}的照片",
    lambda c: f"{c}的重现",
    lambda c: f"一张漂亮的{c}的照片",
    lambda c: f"一张奇怪的{c}的照片",
    lambda c: f"模糊的{c}的照片",
    lambda c: f"卡通{c}",
    lambda c: f"{c}的艺术作品",
    lambda c: f"{c}的素描",
    lambda c: f"刺绣{c}",
    lambda c: f"{c}的像素照",
    lambda c: f"{c}的拍照",
    lambda c: f"{c}的损坏的照片",
    lambda c: f"高质量的{c}的照片",
    lambda c: f"毛绒玩具{c}",
    lambda c: f"漂亮的{c}的照片",
    lambda c: f"小{c}的照片",
    lambda c: f"照片是奇怪的{c}",
    lambda c: f"漫画{c}",
    lambda c: f"{c}的艺术照",
    lambda c: f"{c}的图形",
    lambda c: f"大{c}的照片",
    lambda c: f"黑白的{c}的照片",
    lambda c: f"{c}毛绒玩具",
    lambda c: f"一张{c}的深色照片",
    lambda c: f"{c}的摄影图",
    lambda c: f"{c}的涂鸦照",
    lambda c: f"玩具形状的{c}",
    lambda c: f"拍了{c}的照片",
    lambda c: f"酷酷的{c}的照片",
    lambda c: f"照片里的小{c}",
    lambda c: f"{c}的刺青",
    lambda c: f"{c}的可爱的照片",
    lambda c: f"一张{c}可爱的照片",
    lambda c: f"{c}可爱图片",
    lambda c: f"{c}酷炫图片",
    lambda c: f"一张{c}的酷炫的照片",
    lambda c: f"一张{c}的酷炫图片",
    lambda c: f"这是{c}",
    lambda c: f"{c}的好看照片",
    lambda c: f"一张{c}的好看的图片",
    lambda c: f"{c}的好看图片",
    lambda c: f"{c}的照片。",
    lambda c: f"质量差的{c}的照片。",
    lambda c: f"许多{c}的照片。",
    lambda c: f"{c}的雕塑。",
    lambda c: f"难以看到{c}的照片。",
    lambda c: f"{c}的低分辨率照片。",
    lambda c: f"{c}的渲染。",
    lambda c: f"涂鸦{c}。",
    lambda c: f"{c}的糟糕照片。",
    lambda c: f"{c}的裁剪照片。",
    lambda c: f"{c}的纹身。",
    lambda c: f"{c}的刺绣照片。",
    lambda c: f"很难看到{c}的照片。",
    lambda c: f"{c}的明亮照片。",
    lambda c: f"一张干净的{c}的照片。",
    lambda c: f"一张包含{c}的照片。",
    lambda c: f"{c}的深色照片。",
    lambda c: f"{c}的手绘画。",
    lambda c: f"我的{c}的照片。",
    lambda c: f"不自然的{c}的照片。",
    lambda c: f"一张酷的{c}的照片。",
    lambda c: f"{c}的特写照片。",
    lambda c: f"{c}的黑白照片。",
    lambda c: f"一幅{c}的画。",
    lambda c: f"一幅{c}的绘画。",
    lambda c: f"一张{c}的像素照片。",
    lambda c: f"{c}的雕像。",
    lambda c: f"一张{c}的明亮照片。",
    lambda c: f"{c}的裁剪照片。",
    lambda c: f"人造的{c}的照片。",
    lambda c: f"一张关于{c}的照片。",
    lambda c: f"损坏的{c}的jpeg照片。",
    lambda c: f"{c}的模糊照片。",
    lambda c: f"{c}的相片。",
    lambda c: f"一张{c}的好照片。",
    lambda c: f"{c}的渲染照。",
    lambda c: f"视频游戏中的{c}。",
    lambda c: f"一张{c}的照片。",
    lambda c: f"{c}的涂鸦。",
    lambda c: f"{c}的近距离照片。",
    lambda c: f"{c}的折纸。",
    lambda c: f"{c}在视频游戏中。",
    lambda c: f"{c}的草图。",
    lambda c: f"{c}的涂鸦照。",
    lambda c: f"{c}的折纸形状。",
    lambda c: f"低分辨率的{c}的照片。",
    lambda c: f"玩具{c}。",
    lambda c: f"{c}的副本。",
    lambda c: f"{c}的干净的照片。",
    lambda c: f"一张大{c}的照片。",
    lambda c: f"{c}的重现。",
    lambda c: f"一张漂亮的{c}的照片。",
    lambda c: f"一张奇怪的{c}的照片。",
    lambda c: f"模糊的{c}的照片。",
    lambda c: f"卡通{c}。",
    lambda c: f"{c}的艺术作品。",
    lambda c: f"{c}的素描。",
    lambda c: f"刺绣{c}。",
    lambda c: f"{c}的像素照。",
    lambda c: f"{c}的拍照。",
    lambda c: f"{c}的损坏的照片。",
    lambda c: f"高质量的{c}的照片。",
    lambda c: f"毛绒玩具{c}。",
    lambda c: f"漂亮的{c}的照片。",
    lambda c: f"小{c}的照片。",
    lambda c: f"照片是奇怪的{c}。",
    lambda c: f"漫画{c}。",
    lambda c: f"{c}的艺术照。",
    lambda c: f"{c}的图形。",
    lambda c: f"大{c}的照片。",
    lambda c: f"黑白的{c}的照片。",
    lambda c: f"{c}毛绒玩具。",
    lambda c: f"一张{c}的深色照片。",
    lambda c: f"{c}的摄影图。",
    lambda c: f"{c}的涂鸦照。",
    lambda c: f"玩具形状的{c}。",
    lambda c: f"拍了{c}的照片。",
    lambda c: f"酷酷的{c}的照片。",
    lambda c: f"照片里的小{c}。",
    lambda c: f"{c}的刺青。",
    lambda c: f"{c}的可爱的照片。",
    lambda c: f"一张{c}可爱的照片。",
    lambda c: f"{c}可爱图片。",
    lambda c: f"{c}酷炫图片。",
    lambda c: f"一张{c}的酷炫的照片。",
    lambda c: f"一张{c}的酷炫图片。",
    lambda c: f"这是{c}。",
    lambda c: f"{c}的好看照片。",
    lambda c: f"一张{c}的好看的图片。",
    lambda c: f"{c}的好看图片。",
    lambda c: f"一种叫{c}的花的照片",
    lambda c: f"一种叫{c}的食物的照片",
    lambda c: f"{c}的卫星照片"
]

normal_templates = [lambda c: f"{c}的图片"]

flower_templates = [
    lambda c: f"一种叫{c}的花的照片",
    lambda c: f"一种叫{c}的花卉的照片",
    lambda c: f"一种叫{c}的花朵的照片",
    lambda c: f"一种叫{c}的鲜花的照片",
    lambda c: f"一种叫{c}的花的高清图",
    lambda c: f"一种叫{c}的花卉的高清图",
    lambda c: f"一种叫{c}的花朵的高清图",
    lambda c: f"一种叫{c}的鲜花的高清图",
    lambda c: f"一种叫{c}的花的模糊图片",
    lambda c: f"一种叫{c}的花朵的模糊图片",
    lambda c: f"一种叫{c}的花卉的模糊图片",
    lambda c: f"一种叫{c}的鲜花的模糊图片",
    lambda c: f"一种叫{c}的花的缩放图片",
    lambda c: f"一种叫{c}的花朵的缩放图片",
    lambda c: f"一种叫{c}的花卉的缩放图片",
    lambda c: f"一种叫{c}的鲜花的缩放图片",
    lambda c: f"一种叫{c}的花的摄影图",
    lambda c: f"一种叫{c}的花卉的摄影图",
    lambda c: f"一种叫{c}的花朵的摄影图",
    lambda c: f"一种叫{c}的鲜花的摄影图",
    lambda c: f"一种叫{c}的花的近距离照片",
    lambda c: f"一种叫{c}的花朵的近距离照片",
    lambda c: f"一种叫{c}的花卉的近距离照片",
    lambda c: f"一种叫{c}的鲜花的近距离照片",
    lambda c: f"一种叫{c}的花的裁剪照片",
    lambda c: f"一种叫{c}的花朵的裁剪照片",
    lambda c: f"一种叫{c}的花卉的裁剪照片",
    lambda c: f"一种叫{c}的鲜花的裁剪照片",
    lambda c: f"一种叫{c}的花的好看的图片",
    lambda c: f"一种叫{c}的花朵的好看的图片",
    lambda c: f"一种叫{c}的花卉的好看的图片",
    lambda c: f"一种叫{c}的鲜花的好看的图片",
]

food_templates = [
    lambda c: f"一种叫{c}的食物的照片",
    lambda c: f"一种叫{c}的美食的照片",
    lambda c: f"一种叫{c}的菜的照片",
    lambda c: f"一种叫{c}的食物的高清图",
    lambda c: f"一种叫{c}的美食的高清图",
    lambda c: f"一种叫{c}的菜的高清图",
    lambda c: f"一种叫{c}的食物的模糊图片",
    lambda c: f"一种叫{c}的美食的模糊图片",
    lambda c: f"一种叫{c}的菜的模糊图片",
    lambda c: f"一种叫{c}的食物的缩放图片",
    lambda c: f"一种叫{c}的美食的缩放图片",
    lambda c: f"一种叫{c}的菜的缩放图片",
    lambda c: f"一种叫{c}的食物的摄影图",
    lambda c: f"一种叫{c}的美食的摄影图",
    lambda c: f"一种叫{c}的菜的摄影图",
    lambda c: f"一种叫{c}的食物的近距离照片",
    lambda c: f"一种叫{c}的美食的近距离照片",
    lambda c: f"一种叫{c}的菜的近距离照片",
    lambda c: f"一种叫{c}的食物的裁剪照片",
    lambda c: f"一种叫{c}的美食的裁剪照片",
    lambda c: f"一种叫{c}的菜的裁剪照片",
]

aircraft_templates = [
    lambda c: f"{c}，飞机的照片",
    lambda c: f"{c}，飞机的高清图",
    lambda c: f"{c}，飞机的模糊图片",
    lambda c: f"{c}，飞机的缩放图片",
    lambda c: f"{c}，飞机的摄影图",
    lambda c: f"{c}，战斗机的照片",
    lambda c: f"{c}，战斗机的高清图",
    lambda c: f"{c}，战斗机的模糊图片",
    lambda c: f"{c}，战斗机的缩放图片",
    lambda c: f"{c}，战斗机的摄影图",
    lambda c: f"{c}，老飞机的照片",
    lambda c: f"{c}，老飞机的高清图",
    lambda c: f"{c}，老飞机的模糊图片",
    lambda c: f"{c}，老飞机的缩放图片",
    lambda c: f"{c}，老飞机的摄影图",
    lambda c: f"{c}，大飞机的照片",
    lambda c: f"{c}，大飞机的高清图",
    lambda c: f"{c}，大飞机的模糊图片",
    lambda c: f"{c}，大飞机的缩放图片",
    lambda c: f"{c}，大飞机的摄影图",
    lambda c: f"{c}，小飞机的照片",
    lambda c: f"{c}，小飞机的高清图",
    lambda c: f"{c}，小飞机的模糊图片",
    lambda c: f"{c}，小飞机的缩放图片",
    lambda c: f"{c}，小飞机的摄影图",
    lambda c: f"{c}，军用飞机的照片",
    lambda c: f"{c}，军用飞机的高清图",
    lambda c: f"{c}，军用飞机的模糊图片",
    lambda c: f"{c}，军用飞机的缩放图片",
    lambda c: f"{c}，军用飞机的摄影图",
    lambda c: f"{c}，运输机的照片",
    lambda c: f"{c}，运输机的高清图",
    lambda c: f"{c}，运输机的模糊图片",
    lambda c: f"{c}，运输机的缩放图片",
    lambda c: f"{c}，运输机的摄影图",
    lambda c: f"{c}，公务机的照片",
    lambda c: f"{c}，公务机的高清图",
    lambda c: f"{c}，公务机的模糊图片",
    lambda c: f"{c}，公务机的缩放图片",
    lambda c: f"{c}，公务机的摄影图",
    lambda c: f"{c}，客机的照片",
    lambda c: f"{c}，客机的高清图",
    lambda c: f"{c}，客机的模糊图片",
    lambda c: f"{c}，客机的缩放图片",
    lambda c: f"{c}，客机的摄影图",
    lambda c: f"{c}，喷气机的照片",
    lambda c: f"{c}，喷气机的高清图",
    lambda c: f"{c}，喷气机的模糊图片",
    lambda c: f"{c}，喷气机的缩放图片",
    lambda c: f"{c}，喷气机的摄影图",
    lambda c: f"一种叫{c}的飞机的照片",
    lambda c: f"一种叫{c}的飞机的高清图",
    lambda c: f"一种叫{c}的飞机的模糊图片",
    lambda c: f"一种叫{c}的飞机的缩放图片",
    lambda c: f"一种叫{c}的飞机的摄影图",
    lambda c: f"一种叫{c}的战斗机的照片",
    lambda c: f"一种叫{c}的战斗机的高清图",
    lambda c: f"一种叫{c}的战斗机的模糊图片",
    lambda c: f"一种叫{c}的战斗机的缩放图片",
    lambda c: f"一种叫{c}的战斗机的摄影图",
    lambda c: f"一种叫{c}的老飞机的照片",
    lambda c: f"一种叫{c}的老飞机的高清图",
    lambda c: f"一种叫{c}的老飞机的模糊图片",
    lambda c: f"一种叫{c}的老飞机的缩放图片",
    lambda c: f"一种叫{c}的老飞机的摄影图",
    lambda c: f"一种叫{c}的大飞机的照片",
    lambda c: f"一种叫{c}的大飞机的高清图",
    lambda c: f"一种叫{c}的大飞机的模糊图片",
    lambda c: f"一种叫{c}的大飞机的缩放图片",
    lambda c: f"一种叫{c}的大飞机的摄影图",
    lambda c: f"一种叫{c}的小飞机的照片",
    lambda c: f"一种叫{c}的小飞机的高清图",
    lambda c: f"一种叫{c}的小飞机的模糊图片",
    lambda c: f"一种叫{c}的小飞机的缩放图片",
    lambda c: f"一种叫{c}的小飞机的摄影图",
    lambda c: f"一种叫{c}的军用飞机的照片",
    lambda c: f"一种叫{c}的军用飞机的高清图",
    lambda c: f"一种叫{c}的军用飞机的模糊图片",
    lambda c: f"一种叫{c}的军用飞机的缩放图片",
    lambda c: f"一种叫{c}的军用飞机的摄影图",
    lambda c: f"一种叫{c}的运输机的照片",
    lambda c: f"一种叫{c}的运输机的高清图",
    lambda c: f"一种叫{c}的运输机的模糊图片",
    lambda c: f"一种叫{c}的运输机的缩放图片",
    lambda c: f"一种叫{c}的运输机的摄影图",
    lambda c: f"一种叫{c}的公务机的照片",
    lambda c: f"一种叫{c}的公务机的高清图",
    lambda c: f"一种叫{c}的公务机的模糊图片",
    lambda c: f"一种叫{c}的公务机的缩放图片",
    lambda c: f"一种叫{c}的公务机的摄影图",
    lambda c: f"一种叫{c}的客机的照片",
    lambda c: f"一种叫{c}的客机的高清图",
    lambda c: f"一种叫{c}的客机的模糊图片",
    lambda c: f"一种叫{c}的客机的缩放图片",
    lambda c: f"一种叫{c}的客机的摄影图",
    lambda c: f"一种叫{c}的喷气机的照片",
    lambda c: f"一种叫{c}的喷气机的高清图",
    lambda c: f"一种叫{c}的喷气机的模糊图片",
    lambda c: f"一种叫{c}的喷气机的缩放图片",
    lambda c: f"一种叫{c}的喷气机的摄影图",
]

eurosat_templates = [
    lambda c: f"一张{c}的卫星照片",
    lambda c: f"{c}的卫星照片",
    lambda c: f"一张{c}的高清卫星照片",
    lambda c: f"{c}的高清卫星照片",
    lambda c: f"一张{c}的清晰的卫星照片",
    lambda c: f"{c}的清晰的卫星照片",
    lambda c: f"一张{c}的高质量的卫星照片",
    lambda c: f"{c}的高质量的卫星照片",
    lambda c: f"一张{c}的卫星图",
    lambda c: f"{c}的卫星图",
    lambda c: f"一张{c}的高清卫星图",
    lambda c: f"{c}的高清卫星图",
    lambda c: f"一张{c}的清晰的卫星图",
    lambda c: f"{c}的清晰的卫星图",
    lambda c: f"一张{c}的高质量的卫星图",
    lambda c: f"{c}的高质量的卫星图",
    lambda c: f"一张{c}的卫星图片",
    lambda c: f"{c}的卫星图片",
    lambda c: f"一张{c}的高清卫星图片",
    lambda c: f"{c}的高清卫星图片",
    lambda c: f"一张{c}的清晰的卫星图片",
    lambda c: f"{c}的清晰的卫星图片",
    lambda c: f"一张{c}的高质量的卫星图片",
    lambda c: f"{c}的高质量的卫星图片",
]


hatefulmemes_templates = [
    lambda c: f"一个{c}",
    lambda c: f"{c}",
]

kitti_templates = [
    lambda c: f"照片里{c}",
    lambda c: f"图片里{c}",
    lambda c: f"{c}",
]

cars_templates = [
    lambda c: f"一张{c}的照片",
    lambda c: f"一张我的{c}的照片",
    lambda c: f"我爱我的{c}",
    lambda c: f"一张我肮脏的{c}的照片",
    lambda c: f"一张我干净的{c}的照片",
    lambda c: f"一张我新买的{c}的照片",
    lambda c: f"一张我旧的{c}的照片",
]

dtd_templates = [
    lambda c: f"一张{c}纹理的照片",
    lambda c: f"一张{c}图案的照片",
    lambda c: f"一张{c}物体的照片",
    lambda c: f"一张{c}纹理的图片",
    lambda c: f"一张{c}图案的图片",
    lambda c: f"一张{c}物体的图片",
]

country211_templates = [
    lambda c: f"一张在{c}拍的照片",
    lambda c: f"一张在{c}旅行时拍的照片",
    lambda c: f"一张我家乡{c}的照片",
    lambda c: f"一张展示{c}风光的照片",
]

patch_templates = [
    lambda c: f"一张{c}的医疗照片",
    lambda c: f"一张{c}的ct照片",
    lambda c: f"一张{c}的化验照片",
]

pet_templates = [
    lambda c: f"一种叫{c}的宠物的照片",
    lambda c: f"一种叫{c}的宠物的图片",
    lambda c: f"一种叫{c}的宠物的可爱图片",
    lambda c: f"一种叫{c}的宠物的高清图片",
    lambda c: f"一种叫{c}的宠物的模糊图片",
    lambda c: f"一种叫{c}的宠物的特写照片",
]

cifar100_templates = [
    lambda c: f"一张{c}的照片",
    lambda c: f"一张{c}的模糊照片",
    lambda c: f"一张{c}",
    lambda c: f"一张{c}的低对比度照片",
    lambda c: f"一张{c}的高对比度照片",
    lambda c: f"一张{c}的好照片",
    lambda c: f"一张小{c}的照片",
    lambda c: f"一张大{c}的照片",
    lambda c: f"一张{c}的黑白照片",
    lambda c: f"一张{c}的低对比度的照片",
    lambda c: f"一张{c}的高对比度的照片",
]

caltech101_templates = [
    lambda c: f"{c}的照片",
    lambda c: f"{c}的绘画",
    lambda c: f"{c}的塑料",
    lambda c: f"{c}的雕像",
    lambda c: f"{c}的草图",
    lambda c: f"{c}的刺青",
    lambda c: f"{c}的玩具",
    lambda c: f"{c}的演绎",
    lambda c: f"{c}的装饰",
    lambda c: f"{c}的卡通画",
    lambda c: f"{c}在游戏中",
    lambda c: f"一个豪华的{c}.",
    lambda c: f"{c}的折纸",
    lambda c: f"{c}的艺术画",
    lambda c: f"{c}的涂鸦画",
    lambda c: f"{c}的画",
]

fer_templates = [
    lambda c: f"一张表情{c}的照片",
    lambda c: f"一张表达{c}情绪的照片",
    lambda c: f"一张看起来很{c}的照片",
    lambda c: f"他的脸看起来{c}",
    lambda c: f"他们看起来很{c}",
]