# 我们的测试集在各种情况的预测结果可视化分析：

1. 光照情况
   基本上预测点都偏下偏中，除了xgaze（相对来说分布得更散），由于各个光照条件下的预测值分布没有较大差异，以下以各个模型下的白天室内背面光为例:
![Alt text](Ours_%E7%99%BD%E5%A4%A9%E5%AE%A4%E5%86%85%E8%83%8C%E9%9D%A2%E5%85%89_x_vs_y.jpg)
![Alt text](Columbia_%E7%99%BD%E5%A4%A9%E5%AE%A4%E5%86%85%E8%83%8C%E9%9D%A2%E5%85%89_x_vs_y.jpg)
![Alt text](EVE_%E7%99%BD%E5%A4%A9%E5%AE%A4%E5%86%85%E8%83%8C%E9%9D%A2%E5%85%89_x_vs_y.jpg)
![Alt text](Gaze360_%E7%99%BD%E5%A4%A9%E5%AE%A4%E5%86%85%E8%83%8C%E9%9D%A2%E5%85%89_x_vs_y.jpg)
![Alt text](MPII_%E7%99%BD%E5%A4%A9%E5%AE%A4%E5%86%85%E8%83%8C%E9%9D%A2%E5%85%89_x_vs_y.jpg)
![Alt text](xgaze_%E7%99%BD%E5%A4%A9%E5%AE%A4%E5%86%85%E8%83%8C%E9%9D%A2%E5%85%89_x_vs_y.jpg)

2. 佩戴眼镜
   相对与不戴眼镜来说，戴眼镜时会有更多预测点偏上
![Alt text](Ours_glass_x_vs_y.jpg)
![Alt text](Ours_noglass_x_vs_y.jpg)

3. 头部姿势
   暂未发现差异
   ![Alt text](Ours_upright_x_vs_y.jpg)
   ![Alt text](Ours_not_upright_x_vs_y.jpg)